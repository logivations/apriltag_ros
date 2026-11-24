// detector.clip_max: a tag in direct sunlight has its black cells above the clamp value,
// so the clamped copy is uniformly grey there and only the original image finds the tag
// (Würth platform marker 151, 2026-11-23, black cells 124-154 with clip_max 140).

#include <apriltag.h>
#include <tag36h11.h>

#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#include <class_loader/class_loader.hpp>
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/node_factory.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

constexpr int kWidth = 848;
constexpr int kHeight = 480;

// draw tag36h11 `id` with its top left corner at (x0, y0), `scale` pixels per tag cell
static void draw_tag(std::vector<uint8_t>& img, apriltag_family_t* family, int id, int x0, int y0, int scale, uint8_t black, uint8_t white)
{
    image_u8_t* tag = apriltag_to_image(family, id);
    for(int y = 0; y < tag->height * scale; y++) {
        for(int x = 0; x < tag->width * scale; x++) {
            const bool is_white = tag->buf[(y / scale) * tag->stride + x / scale] > 127;
            img[(y0 + y) * kWidth + x0 + x] = is_white ? white : black;
        }
    }
    image_u8_destroy(tag);
}

TEST(ClipMax, FindsTagWhoseBlackCellsAreAboveTheClampValueOnce)
{
    // 152 has normal contrast and is found in both images, 151 is in sunlight
    std::vector<uint8_t> pixels(kWidth * kHeight, 120);
    apriltag_family_t* family = tag36h11_create();
    draw_tag(pixels, family, 152, 100, 160, 16, 30, 230);
    draw_tag(pixels, family, 151, 500, 160, 16, 150, 250);
    tag36h11_destroy(family);

    class_loader::ClassLoader loader(APRILTAG_NODE_LIBRARY);
    auto factory = loader.createInstance<rclcpp_components::NodeFactory>("rclcpp_components::NodeFactoryTemplate<AprilTagNode>");
    rclcpp::NodeOptions options;
    options.arguments({"--ros-args", "-r", "__ns:=/clip_max_test"});
    options.parameter_overrides({{"detector.clip_max", 140}});
    auto apriltag = factory->create_node_instance(options);

    auto tester = std::make_shared<rclcpp::Node>("tester", "/clip_max_test");
    auto pub_image = tester->create_publisher<sensor_msgs::msg::Image>("image_rect", 1);
    auto pub_info = tester->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", 1);
    std::vector<apriltag_msgs::msg::AprilTagDetectionArray> received;
    auto sub = tester->create_subscription<apriltag_msgs::msg::AprilTagDetectionArray>(
        "detections", 10, [&received](const apriltag_msgs::msg::AprilTagDetectionArray& msg) { received.push_back(msg); });
    auto enable = tester->create_client<std_srvs::srv::SetBool>("apriltag/toggle_enabled");

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(apriltag.get_node_base_interface());
    executor.add_node(tester);

    ASSERT_TRUE(enable->wait_for_service(5s));
    auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
    request->data = true;
    auto response = enable->async_send_request(request);
    ASSERT_EQ(executor.spin_until_future_complete(response, 5s), rclcpp::FutureReturnCode::SUCCESS);

    sensor_msgs::msg::CameraInfo info;
    info.width = kWidth;
    info.height = kHeight;
    info.k = {606.0, 0.0, 417.0, 0.0, 606.0, 240.0, 0.0, 0.0, 1.0};
    info.p = {606.0, 0.0, 417.0, 0.0, 0.0, 606.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0};

    sensor_msgs::msg::Image image;
    image.width = kWidth;
    image.height = kHeight;
    image.encoding = "mono8";
    image.step = kWidth;
    image.data = pixels;

    // the node drops images until it has a camera info, so publish both until it answers
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while(received.empty() && std::chrono::steady_clock::now() < deadline) {
        pub_info->publish(info);
        pub_image->publish(image);
        for(int i = 0; i < 10 && received.empty(); i++) {
            executor.spin_some();
            std::this_thread::sleep_for(10ms);
        }
    }
    ASSERT_FALSE(received.empty()) << "no detections published";

    std::vector<int> ids;
    for(const auto& detection : received.front().detections) {
        ids.push_back(detection.id);
    }
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(ids, (std::vector<int>{151, 152}));
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    const int result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
