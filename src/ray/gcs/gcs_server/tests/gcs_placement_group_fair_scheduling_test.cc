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

// Tests for fair weighted GPU-memory×Time scheduling in GcsPlacementGroupManager.

#include <memory>
#include <utility>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
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

static rpc::CreatePlacementGroupRequest GenGpuPlacementGroupRequest(
    const std::string &name,
    int bundles_count,
    double gpu_num,
    const JobID &job_id,
    const std::string &selector_resource = "") {
  rpc::CreatePlacementGroupRequest request;
  std::vector<std::unordered_map<std::string, double>> bundles;
  std::unordered_map<std::string, double> bundle;
  bundle["GPU"] = gpu_num;
  bundle["CPU"] = 1.0;
  if (!selector_resource.empty()) {
    bundle[selector_resource] = gpu_num;
  }
  for (int i = 0; i < bundles_count; ++i) {
    bundles.push_back(bundle);
  }
  auto pg_spec = Mocker::GenPlacementGroupCreation(
      name, bundles, rpc::PlacementStrategy::SPREAD, job_id, ActorID::Nil());
  request.mutable_placement_group_spec()->CopyFrom(pg_spec.GetMessage());
  return request;
}

static SchedulePgRequest *FindScheduledRequestForJob(
    std::vector<SchedulePgRequest> &requests, const JobID &job_id) {
  for (auto &request : requests) {
    if (request.placement_group->GetCreatorJobId() == job_id) {
      return &request;
    }
  }
  return nullptr;
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

TEST_F(GcsFairSchedulingTest, GpuPlacementGroupRegistration) {
  JobID job1 = JobID::FromInt(1);

  auto req = GenGpuPlacementGroupRequest("pg1", 2, 4.0, job1);
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

  ASSERT_EQ(request.placement_group, pg);
  ASSERT_EQ(request.placement_group->GetCreatorJobId(), job1);
  ASSERT_EQ(pg->GetState(), rpc::PlacementGroupTableData::PENDING);
}

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

  gcs_placement_group_manager_->RegisterPlacementGroup(pg1, cb);
  std::move(*put_cb).Post("MultipleJobsRegisterPgs_1", true);
  io_context_.poll();

  ASSERT_GE(scheduled_pgs.size(), 1);
  ASSERT_EQ(scheduled_pgs[0]->GetCreatorJobId(), job1);

  gcs_placement_group_manager_->RegisterPlacementGroup(pg2, cb);
  std::move(*put_cb).Post("MultipleJobsRegisterPgs_2", true);
  io_context_.poll();

  ASSERT_GE(scheduled_pgs.size(), 1);
  ASSERT_NE(pg1->GetCreatorJobId(), pg2->GetCreatorJobId());
}

TEST_F(GcsFairSchedulingTest, BundlesContainGpuResources) {
  JobID job1 = JobID::FromInt(1);

  auto req = GenGpuPlacementGroupRequest("pg1", 2, 8.0, job1);
  auto pg = std::make_shared<GcsPlacementGroup>(req, "", counter_);

  const auto &bundles = pg->GetPlacementGroupTableData().bundles();
  ASSERT_EQ(bundles.size(), 2);

  for (const auto &bundle : bundles) {
    auto it = bundle.unit_resources().find("GPU");
    ASSERT_NE(it, bundle.unit_resources().end());
    ASSERT_DOUBLE_EQ(it->second, 8.0);
  }
}

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

  ASSERT_EQ(request.placement_group, pg);

  request.failure_callback(pg, true);

  ASSERT_EQ(pg->GetState(), rpc::PlacementGroupTableData::PENDING);
  ASSERT_GE(pg->GetStats().scheduling_attempt(), 1);
}

TEST_F(GcsFairSchedulingTest, CpuOnlyPlacementGroupRegistration) {
  JobID job1 = JobID::FromInt(1);

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

  ASSERT_EQ(request.placement_group, pg);

  const auto &bundles = pg->GetPlacementGroupTableData().bundles();
  for (const auto &bundle : bundles) {
    auto it = bundle.unit_resources().find("GPU");
    ASSERT_EQ(it, bundle.unit_resources().end());
  }
}

TEST_F(GcsFairSchedulingTest, BundleMemoryTimeWeighting) {
  gcs_placement_group_manager_->ResetGpuUsageForTesting();

  JobID job1 = JobID::FromInt(101);
  auto pg_5090 = std::make_shared<GcsPlacementGroup>(
      GenGpuPlacementGroupRequest("pg-5090", 1, 1.0, job1, "gpu_5090"), "", counter_);
  auto pg_pro6000 = std::make_shared<GcsPlacementGroup>(
      GenGpuPlacementGroupRequest("pg-pro6000", 1, 1.0, job1, "gpu_pro6000"),
      "",
      counter_);
  auto pg_generic = std::make_shared<GcsPlacementGroup>(
      GenGpuPlacementGroupRequest("pg-generic", 1, 2.0, job1), "", counter_);

  ASSERT_DOUBLE_EQ(
      gcs_placement_group_manager_->GetBundleMemoryTimeUnitsForTesting(
          pg_5090->GetPlacementGroupTableData().bundles(0)),
      1.0);
  ASSERT_DOUBLE_EQ(
      gcs_placement_group_manager_->GetBundleMemoryTimeUnitsForTesting(
          pg_pro6000->GetPlacementGroupTableData().bundles(0)),
      3.0);
  ASSERT_DOUBLE_EQ(
      gcs_placement_group_manager_->GetBundleMemoryTimeUnitsForTesting(
          pg_generic->GetPlacementGroupTableData().bundles(0)),
      2.0);
}

TEST_F(GcsFairSchedulingTest,
       WeightedMemoryTimeFairSchedulingPrefersLowerUsageJob) {
  gcs_placement_group_manager_->ResetGpuUsageForTesting();

  JobID heavy_job = JobID::FromInt(201);
  JobID light_job = JobID::FromInt(202);

  auto running_heavy_pg = std::make_shared<GcsPlacementGroup>(
      GenGpuPlacementGroupRequest(
          "running-heavy", 1, 1.0, heavy_job, "gpu_pro6000"),
      "",
      counter_);
  auto running_light_pg = std::make_shared<GcsPlacementGroup>(
      GenGpuPlacementGroupRequest(
          "running-light", 1, 1.0, light_job, "gpu_5090"),
      "",
      counter_);

  std::unique_ptr<Postable<void(bool)>> put_cb;
  EXPECT_CALL(*store_client_, AsyncPut(_, _, _, _, _))
      .Times(AtLeast(2))
      .WillRepeatedly(DoAll(SaveArgToUniquePtr<4>(&put_cb), Return(Status::OK())));

  std::vector<SchedulePgRequest> scheduled_requests;
  EXPECT_CALL(*gcs_placement_group_scheduler_, ScheduleUnplacedBundles(_))
      .Times(2)
      .WillRepeatedly(Invoke([&scheduled_requests](const SchedulePgRequest &request) {
        scheduled_requests.push_back(request);
      }));

  auto cb = [](Status s) {};
  gcs_placement_group_manager_->RegisterPlacementGroup(running_heavy_pg, cb);
  ASSERT_NE(put_cb, nullptr);
  std::move(*put_cb).Post("WeightedMemoryTime_running_heavy", true);
  io_context_.poll();

  ASSERT_EQ(scheduled_requests.size(), 1);
  auto heavy_request = FindScheduledRequestForJob(scheduled_requests, heavy_job);
  ASSERT_NE(heavy_request, nullptr);
  running_heavy_pg->GetMutableBundle(0)->set_node_id(NodeID::FromRandom().Binary());
  running_heavy_pg->UpdateState(rpc::PlacementGroupTableData::PREPARED);
  heavy_request->success_callback(running_heavy_pg);
  io_context_.poll();

  gcs_placement_group_manager_->RegisterPlacementGroup(running_light_pg, cb);
  ASSERT_NE(put_cb, nullptr);
  std::move(*put_cb).Post("WeightedMemoryTime_running_light", true);
  io_context_.poll();

  ASSERT_EQ(scheduled_requests.size(), 2);
  auto light_request = FindScheduledRequestForJob(scheduled_requests, light_job);
  ASSERT_NE(light_request, nullptr);
  running_light_pg->GetMutableBundle(0)->set_node_id(NodeID::FromRandom().Binary());
  running_light_pg->UpdateState(rpc::PlacementGroupTableData::PREPARED);
  light_request->success_callback(running_light_pg);
  io_context_.poll();

  Mock::VerifyAndClearExpectations(gcs_placement_group_scheduler_.get());
  Mock::VerifyAndClearExpectations(store_client_.get());

  gcs_placement_group_manager_->UpdateGpuUsageForTesting();
  absl::SleepFor(absl::Milliseconds(20));
  gcs_placement_group_manager_->UpdateGpuUsageForTesting();

  const double heavy_score =
      gcs_placement_group_manager_->GetJobMemoryTimeUsageScoreForTesting(heavy_job);
  const double light_score =
      gcs_placement_group_manager_->GetJobMemoryTimeUsageScoreForTesting(light_job);
  ASSERT_GT(heavy_score, light_score * 2.5);

  std::unique_ptr<Postable<void(bool)>> pending_put_cb;
  EXPECT_CALL(*store_client_, AsyncPut(_, _, _, _, _))
      .Times(AtLeast(2))
      .WillRepeatedly(
          DoAll(SaveArgToUniquePtr<4>(&pending_put_cb), Return(Status::OK())));

  SchedulePgRequest chosen_request;
  EXPECT_CALL(*gcs_placement_group_scheduler_, ScheduleUnplacedBundles(_))
      .WillOnce(DoAll(SaveArg<0>(&chosen_request)));

  auto pending_heavy_pg = std::make_shared<GcsPlacementGroup>(
      GenGpuPlacementGroupRequest(
          "pending-heavy", 1, 1.0, heavy_job, "gpu_pro6000"),
      "",
      counter_);
  auto pending_light_pg = std::make_shared<GcsPlacementGroup>(
      GenGpuPlacementGroupRequest(
          "pending-light", 1, 1.0, light_job, "gpu_5090"),
      "",
      counter_);

  gcs_placement_group_manager_->RegisterPlacementGroup(pending_heavy_pg, cb);
  gcs_placement_group_manager_->RegisterPlacementGroup(pending_light_pg, cb);

  gcs_placement_group_manager_->SchedulePendingPlacementGroups();

  ASSERT_EQ(chosen_request.placement_group->GetCreatorJobId(), light_job);
}

}  // namespace gcs
}  // namespace ray
