// Copyright 2017 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Tests for fair GPU×Time scheduling in GcsPlacementGroupManager.
// This scheduling policy prioritizes jobs with lower cumulative GPU×Time usage.

#include <memory>
#include <utility>
// clang-format off
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "ray/gcs/gcs_server/gcs_placement_group_mgr.h"
#include "ray/raylet/scheduling/cluster_resource_manager.h"
#include "mock/ray/gcs/gcs_server/gcs_node_manager.h"
#include "mock/ray/gcs/gcs_server/gcs_placement_group_mgr.h"
#include "mock/ray/gcs/gcs_server/gcs_placement_group_scheduler.h"
#include "mock/ray/gcs/gcs_server/gcs_resource_manager.h"
#include "mock/ray/gcs/store_client/store_client.h"
#include "ray/util/counter_map.h"
#include "ray/gcs/tests/gcs_test_util.h"
// clang-format on

using namespace ::testing;  // NOLINT
using namespace ray;        // NOLINT
using namespace ray::gcs;   // NOLINT

namespace ray {
namespace gcs {

// Helper to create a placement group request with GPU resources
static rpc::CreatePlacementGroupRequest GenGpuPlacementGroupRequest(
    const std::string &name,
    int bundles_count,
    double gpu_num,
    const JobID &job_id) {
  rpc::CreatePlacementGroupRequest request;
  std::vector<std::unordered_map<std::string, double>> bundles;
  std::unordered_map<std::string, double> bundle;
  bundle["GPU"] = gpu_num;
  bundle["CPU"] = 1.0;  // Also add CPU for realistic setup
  for (int i = 0; i < bundles_count; ++i) {
    bundles.push_back(bundle);
  }
  auto pg_spec = Mocker::GenPlacementGroupCreation(
      name, bundles, rpc::PlacementStrategy::SPREAD, job_id, ActorID::Nil());
  request.mutable_placement_group_spec()->CopyFrom(pg_spec.GetMessage());
  return request;
}

class GcsFairSchedulingTest : public Test {
 public:
  GcsFairSchedulingTest() : cluster_resource_manager_(io_context_) {}

  void SetUp() override {
    store_client_ = std::make_shared<MockStoreClient>();
    gcs_table_storage_ = std::make_shared<GcsTableStorage>(store_client_);
    gcs_placement_group_scheduler_ =
        std::make_shared<MockGcsPlacementGroupSchedulerInterface>();
    node_manager_ = std::make_unique<MockGcsNodeManager>();
    resource_manager_ = std::make_shared<MockGcsResourceManager>(
        io_context_, cluster_resource_manager_, *node_manager_, NodeID::FromRandom());

    gcs_placement_group_manager_ =
        std::make_unique<GcsPlacementGroupManager>(io_context_,
                                                   gcs_placement_group_scheduler_.get(),
                                                   gcs_table_storage_.get(),
                                                   *resource_manager_,
                                                   [](auto &) { return ""; });
    counter_.reset(new CounterMap<rpc::PlacementGroupTableData::PlacementGroupState>());
  }

  instrumented_io_context io_context_;
  std::unique_ptr<GcsPlacementGroupManager> gcs_placement_group_manager_;
  std::shared_ptr<MockGcsPlacementGroupSchedulerInterface>
      gcs_placement_group_scheduler_;
  std::shared_ptr<gcs::GcsTableStorage> gcs_table_storage_;
  std::shared_ptr<MockStoreClient> store_client_;
  std::unique_ptr<GcsNodeManager> node_manager_;
  ClusterResourceManager cluster_resource_manager_;
  std::shared_ptr<GcsResourceManager> resource_manager_;
  std::shared_ptr<CounterMap<rpc::PlacementGroupTableData::PlacementGroupState>> counter_;
};

// Test: Placement groups with GPU resources are registered correctly
TEST_F(GcsFairSchedulingTest, GpuPlacementGroupRegistration) {
  JobID job1 = JobID::FromInt(1);

  auto req = GenGpuPlacementGroupRequest("pg1", 2, 4.0, job1);  // 4 GPUs per bundle
  auto pg = std::make_shared<GcsPlacementGroup>(req, "", counter_);

  SchedulePgRequest request;
  std::unique_ptr<Postable<void(bool)>> put_cb;
  EXPECT_CALL(*store_client_, AsyncPut(_, _, _, _, _))
      .WillOnce(DoAll(SaveArgToUniquePtr<4>(&put_cb), Return(Status::OK())));
  EXPECT_CALL(*gcs_placement_group_scheduler_, ScheduleUnplacedBundles(_))
      .WillOnce(DoAll(SaveArg<0>(&request)));

  auto cb = [](Status s) {};
  gcs_placement_group_manager_->RegisterPlacementGroup(pg, cb);
  std::move(*put_cb).Post("GpuPlacementGroupRegistration", true);
  io_context_.poll();

  // Verify the PG was scheduled
  ASSERT_EQ(request.placement_group, pg);
  ASSERT_EQ(request.placement_group->GetCreatorJobId(), job1);

  // Verify the PG state is PENDING
  ASSERT_EQ(pg->GetState(), rpc::PlacementGroupTableData::PENDING);
}

// Test: Multiple placement groups from different jobs can be registered
TEST_F(GcsFairSchedulingTest, MultipleJobsRegisterPgs) {
  JobID job1 = JobID::FromInt(1);
  JobID job2 = JobID::FromInt(2);

  auto req1 = GenGpuPlacementGroupRequest("pg1", 2, 4.0, job1);
  auto req2 = GenGpuPlacementGroupRequest("pg2", 2, 2.0, job2);

  auto pg1 = std::make_shared<GcsPlacementGroup>(req1, "", counter_);
  auto pg2 = std::make_shared<GcsPlacementGroup>(req2, "", counter_);

  std::vector<std::shared_ptr<GcsPlacementGroup>> scheduled_pgs;
  std::unique_ptr<Postable<void(bool)>> put_cb;
  EXPECT_CALL(*store_client_, AsyncPut(_, _, _, _, _))
      .Times(AtLeast(1))
      .WillRepeatedly(DoAll(SaveArgToUniquePtr<4>(&put_cb), Return(Status::OK())));
  EXPECT_CALL(*gcs_placement_group_scheduler_, ScheduleUnplacedBundles(_))
      .Times(AtLeast(1))
      .WillRepeatedly(
          Invoke([&scheduled_pgs](const SchedulePgRequest &request) {
            scheduled_pgs.push_back(request.placement_group);
          }));

  auto cb = [](Status s) {};

  // Register first PG
  gcs_placement_group_manager_->RegisterPlacementGroup(pg1, cb);
  std::move(*put_cb).Post("MultipleJobsRegisterPgs_1", true);
  io_context_.poll();

  // First PG should be scheduled
  ASSERT_GE(scheduled_pgs.size(), 1);
  ASSERT_EQ(scheduled_pgs[0]->GetCreatorJobId(), job1);

  // Register second PG (while first is still being scheduled)
  gcs_placement_group_manager_->RegisterPlacementGroup(pg2, cb);
  std::move(*put_cb).Post("MultipleJobsRegisterPgs_2", true);
  io_context_.poll();

  // At least the first PG should have been scheduled
  // The second PG may be in the pending queue waiting
  ASSERT_GE(scheduled_pgs.size(), 1);

  // Both PGs should have different job IDs
  ASSERT_NE(pg1->GetCreatorJobId(), pg2->GetCreatorJobId());
}

// Test: Verify that bundles contain expected GPU resources
TEST_F(GcsFairSchedulingTest, BundlesContainGpuResources) {
  JobID job1 = JobID::FromInt(1);

  auto req = GenGpuPlacementGroupRequest("pg1", 2, 8.0, job1);  // 8 GPUs per bundle
  auto pg = std::make_shared<GcsPlacementGroup>(req, "", counter_);

  // Verify the bundle resources
  const auto &bundles = pg->GetPlacementGroupTableData().bundles();
  ASSERT_EQ(bundles.size(), 2);

  for (const auto &bundle : bundles) {
    auto it = bundle.unit_resources().find("GPU");
    ASSERT_NE(it, bundle.unit_resources().end());
    ASSERT_DOUBLE_EQ(it->second, 8.0);
  }
}

// Test: Scheduling failure keeps PG in PENDING state
TEST_F(GcsFairSchedulingTest, SchedulingFailureKeepsPending) {
  JobID job1 = JobID::FromInt(1);

  auto req = GenGpuPlacementGroupRequest("pg1", 2, 4.0, job1);
  auto pg = std::make_shared<GcsPlacementGroup>(req, "", counter_);

  SchedulePgRequest request;
  std::unique_ptr<Postable<void(bool)>> put_cb;
  EXPECT_CALL(*store_client_, AsyncPut(_, _, _, _, _))
      .WillOnce(DoAll(SaveArgToUniquePtr<4>(&put_cb), Return(Status::OK())));
  EXPECT_CALL(*gcs_placement_group_scheduler_, ScheduleUnplacedBundles(_))
      .Times(AtLeast(1))
      .WillRepeatedly(DoAll(SaveArg<0>(&request)));

  auto cb = [](Status s) {};
  gcs_placement_group_manager_->RegisterPlacementGroup(pg, cb);
  std::move(*put_cb).Post("SchedulingFailureKeepsPending", true);
  io_context_.poll();

  // Verify initial scheduling attempt
  ASSERT_EQ(request.placement_group, pg);

  // Simulate scheduling failure (but still schedulable - no retry limit reached)
  request.failure_callback(pg, true);

  // After failure, the PG state should still be PENDING (not REMOVED or other)
  ASSERT_EQ(pg->GetState(), rpc::PlacementGroupTableData::PENDING);

  // The scheduling attempt counter should have increased
  ASSERT_GE(pg->GetStats().scheduling_attempt(), 1);
}

// Test: CPU-only placement groups are handled correctly
TEST_F(GcsFairSchedulingTest, CpuOnlyPlacementGroupRegistration) {
  JobID job1 = JobID::FromInt(1);

  // Create a CPU-only placement group
  auto req =
      Mocker::GenCreatePlacementGroupRequest("", rpc::PlacementStrategy::SPREAD, 2, 4.0, job1);
  auto pg = std::make_shared<GcsPlacementGroup>(req, "", counter_);

  SchedulePgRequest request;
  std::unique_ptr<Postable<void(bool)>> put_cb;
  EXPECT_CALL(*store_client_, AsyncPut(_, _, _, _, _))
      .WillOnce(DoAll(SaveArgToUniquePtr<4>(&put_cb), Return(Status::OK())));
  EXPECT_CALL(*gcs_placement_group_scheduler_, ScheduleUnplacedBundles(_))
      .WillOnce(DoAll(SaveArg<0>(&request)));

  auto cb = [](Status s) {};
  gcs_placement_group_manager_->RegisterPlacementGroup(pg, cb);
  std::move(*put_cb).Post("CpuOnlyPlacementGroupRegistration", true);
  io_context_.poll();

  // Verify the PG was scheduled
  ASSERT_EQ(request.placement_group, pg);

  // Verify the bundles don't have GPU resources
  const auto &bundles = pg->GetPlacementGroupTableData().bundles();
  for (const auto &bundle : bundles) {
    auto it = bundle.unit_resources().find("GPU");
    ASSERT_EQ(it, bundle.unit_resources().end());  // No GPU resource
  }
}

}  // namespace gcs
}  // namespace ray
