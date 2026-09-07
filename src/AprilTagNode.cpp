// ros
#include "pose_estimation.hpp"
#include <optional>
#include <apriltag_msgs/msg/april_tag_detection.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#ifdef cv_bridge_HPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <image_transport/camera_subscriber.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <std_srvs/srv/set_bool.hpp>

// apriltag
#include "tag_functions.hpp"
#include <apriltag.h>


#define IF(N, V) \
    if(assign_check(parameter, N, V)) continue;

template<typename T>
void assign(const rclcpp::Parameter& parameter, T& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
void assign(const rclcpp::Parameter& parameter, std::atomic<T>& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
bool assign_check(const rclcpp::Parameter& parameter, const std::string& name, T& var)
{
    if(parameter.get_name() == name) {
        assign(parameter, var);
        return true;
    }
    return false;
}

rcl_interfaces::msg::ParameterDescriptor
descr(const std::string& description, const bool& read_only = false)
{
    rcl_interfaces::msg::ParameterDescriptor descr;

    descr.description = description;
    descr.read_only = read_only;

    return descr;
}

class AprilTagNode : public rclcpp::Node {
public:
    AprilTagNode(const rclcpp::NodeOptions& options);

    ~AprilTagNode() override;

private:
    const OnSetParametersCallbackHandle::SharedPtr cb_parameter;

    apriltag_family_t* tf;
    apriltag_detector_t* const td;

    // parameter
    std::mutex mutex;
    std::atomic<double> tag_edge_size;
    std::atomic<int> max_hamming;
    std::atomic<int> clip_max;
    std::atomic<bool> profile;
    std::atomic<bool> publish_tf;
    std::unordered_map<int, std::string> tag_frames;
    std::unordered_map<int, double> tag_sizes;

    std::function<void(apriltag_family_t*)> tf_destructor;

    rclcpp::Subscription<sensor_msgs::msg::Image>::ConstSharedPtr image_sub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::ConstSharedPtr cam_info_subscriber;
    const rclcpp::Publisher<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr pub_detections;
    tf2_ros::TransformBroadcaster tf_broadcaster;

    pose_estimation_f estimate_pose = nullptr;
    std::optional<sensor_msgs::msg::CameraInfo> camera_info;

    void onImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img);
    void onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci);

    rcl_interfaces::msg::SetParametersResult onParameter(const std::vector<rclcpp::Parameter>& parameters);

    rclcpp::QoS getQoS();

    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr toggle_enabled_srv;

    bool scanning = false;

    void toggle_enabled(
        const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
        std::shared_ptr<std_srvs::srv::SetBool::Response> response);
};

RCLCPP_COMPONENTS_REGISTER_NODE(AprilTagNode)


AprilTagNode::AprilTagNode(const rclcpp::NodeOptions& options)
  : Node("apriltag", options),
    // parameter
    cb_parameter(add_on_set_parameters_callback(std::bind(&AprilTagNode::onParameter, this, std::placeholders::_1))),
    td(apriltag_detector_create()),
    // topics
    image_sub(create_subscription<sensor_msgs::msg::Image>(
      this->get_node_topics_interface()->resolve_topic_name("image_rect"), getQoS(),
      std::bind(&AprilTagNode::onImage, this, std::placeholders::_1))),
    cam_info_subscriber(create_subscription<sensor_msgs::msg::CameraInfo>(
      this->get_node_topics_interface()->resolve_topic_name("camera_info"), rclcpp::QoS{rclcpp::KeepLast(1)}.best_effort(),
      std::bind(&AprilTagNode::onCameraInfo, this, std::placeholders::_1))),
    pub_detections(create_publisher<apriltag_msgs::msg::AprilTagDetectionArray>("detections", rclcpp::QoS(1))),
    tf_broadcaster(this)
{
    toggle_enabled_srv = this->create_service<std_srvs::srv::SetBool>("~/toggle_enabled", std::bind(&AprilTagNode::toggle_enabled, this, std::placeholders::_1, std::placeholders::_2));

    // read-only parameters
    const std::string tag_family = declare_parameter("family", "36h11", descr("tag family", true));
    tag_edge_size = declare_parameter("size", 1.0, descr("default tag size"));

    // get tag names, IDs and sizes
    const auto ids = declare_parameter("tag.ids", std::vector<int64_t>{}, descr("tag ids", true));
    const auto frames = declare_parameter("tag.frames", std::vector<std::string>{}, descr("tag frame names per id", true));
    const auto sizes = declare_parameter("tag.sizes", std::vector<double>{}, descr("tag sizes per id", true));

    // get method for estimating tag pose
    estimate_pose = pose_estimation_methods.at(declare_parameter("pose_estimation_method", "pnp", descr("pose estimation method: \"pnp\" (more accurate) or \"homography\" (faster)", true)));

    // detector parameters in "detector" namespace
    declare_parameter("detector.threads", td->nthreads, descr("number of threads"));
    declare_parameter("detector.decimate", td->quad_decimate, descr("decimate resolution for quad detection"));
    declare_parameter("detector.blur", td->quad_sigma, descr("sigma of Gaussian blur for quad detection"));
    declare_parameter("detector.refine", td->refine_edges, descr("snap to strong gradients"));
    declare_parameter("detector.sharpening", td->decode_sharpening, descr("sharpening of decoded images"));
    declare_parameter("detector.debug", td->debug, descr("write additional debugging images to working directory"));
    // Saturated regions (ceiling lights, windows) next to a tag push the detector's
    // local min/max threshold above the tag's white cells and the quad is lost.
    // Clamping intensities before detection keeps the threshold near the tag's own
    // black/white levels. 0 disables clamping.
    declare_parameter("detector.clip_max", 0, descr("clamp pixel intensities above this value (1-254) before detection, 0 = off"));

    declare_parameter("max_hamming", 0, descr("reject detections with more corrected bits than allowed"));
    declare_parameter("profile", false, descr("print profiling information to stdout"));
    declare_parameter("publish_tf", false, descr("estimate pose and publish tf"));

    if(!frames.empty()) {
        if(ids.size() != frames.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and frames (" + std::to_string(frames.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { tag_frames[ids[i]] = frames[i]; }
    }

    if(!sizes.empty()) {
        // use tag specific size
        if(ids.size() != sizes.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and sizes (" + std::to_string(sizes.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { tag_sizes[ids[i]] = sizes[i]; }
    }

    if(tag_fun.count(tag_family)) {
        tf = tag_fun.at(tag_family).first();
        tf_destructor = tag_fun.at(tag_family).second;
        apriltag_detector_add_family(td, tf);
    }
    else {
        throw std::runtime_error("Unsupported tag family: " + tag_family);
    }
}

AprilTagNode::~AprilTagNode()
{
    apriltag_detector_destroy(td);
    tf_destructor(tf);
}

void AprilTagNode::onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci)
{
    camera_info = *msg_ci;
  }
void AprilTagNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img)
{
    if(!scanning){
        return;
    }
    if (!camera_info.has_value()) {
        RCLCPP_WARN(get_logger(), "No camera info available yet");
        return;
    }

    const sensor_msgs::msg::CameraInfo& ci = camera_info.value();
    // camera intrinsics for rectified images
    const std::array<double, 4> intrinsics = {ci.p.data()[0], ci.p.data()[5], ci.p.data()[2], ci.p.data()[6]};

    // convert to 8bit monochrome image
    cv::Mat img_uint8 = cv_bridge::toCvShare(msg_img, "mono8")->image;

    // optionally clamp saturated pixels (see detector.clip_max); toCvShare may alias the
    // message buffer, so write into a fresh (continuous) matrix
    const int clip = clip_max;
    if(clip > 0 && clip < 255) {
        cv::Mat clipped;
        cv::min(img_uint8, clip, clipped);
        img_uint8 = clipped;
    }
    if(!img_uint8.isContinuous()) {
        img_uint8 = img_uint8.clone();
    }

    image_u8_t im{img_uint8.cols, img_uint8.rows, img_uint8.cols, img_uint8.data};

    // detect tags
    mutex.lock();
    zarray_t* detections = apriltag_detector_detect(td, &im);
    mutex.unlock();

    if(profile)
        timeprofile_display(td->tp);

    apriltag_msgs::msg::AprilTagDetectionArray msg_detections;
    msg_detections.header = msg_img->header;

    std::vector<geometry_msgs::msg::TransformStamped> tfs;

    for(int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        RCLCPP_DEBUG(get_logger(),
                     "detection %3d: id (%2dx%2d)-%-4d, hamming %d, margin %8.3f\n",
                     i, det->family->nbits, det->family->h, det->id,
                     det->hamming, det->decision_margin);

        // ignore untracked tags
        if(!tag_frames.empty() && !tag_frames.count(det->id)) { continue; }

        // reject detections with more corrected bits than allowed
        if(det->hamming > max_hamming) { continue; }

        // detection
        apriltag_msgs::msg::AprilTagDetection msg_detection;
        msg_detection.family = std::string(det->family->name);
        msg_detection.id = det->id;
        msg_detection.hamming = det->hamming;
        msg_detection.decision_margin = det->decision_margin;
        msg_detection.centre.x = det->c[0];
        msg_detection.centre.y = det->c[1];
        std::memcpy(msg_detection.corners.data(), det->p, sizeof(double) * 8);
        std::memcpy(msg_detection.homography.data(), det->H->data, sizeof(double) * 9);
        msg_detections.detections.push_back(msg_detection);

        if(!publish_tf){
            continue;
        }
        // 3D orientation and position
        geometry_msgs::msg::TransformStamped tf;
        tf.header = msg_img->header;
        // set child frame name by generic tag name or configured tag name, prefixed with node name
        std::string base_frame_name = std::to_string(det->id);
        tf.child_frame_id = std::string(this->get_name()) + "/" + base_frame_name;
        const double size = tag_sizes.count(det->id) ? tag_sizes.at(det->id) : tag_edge_size.load();
        if(estimate_pose != nullptr) {
            tf.transform = estimate_pose(det, intrinsics, size);
        }

        tfs.push_back(tf);
    }

    pub_detections->publish(msg_detections);
    if(publish_tf){
        tf_broadcaster.sendTransform(tfs);
    }

    apriltag_detections_destroy(detections);
}

rcl_interfaces::msg::SetParametersResult
AprilTagNode::onParameter(const std::vector<rclcpp::Parameter>& parameters)
{
    rcl_interfaces::msg::SetParametersResult result;

    mutex.lock();

    for(const rclcpp::Parameter& parameter : parameters) {
        RCLCPP_DEBUG_STREAM(get_logger(), "setting: " << parameter);

        IF("detector.threads", td->nthreads)
        IF("detector.decimate", td->quad_decimate)
        IF("detector.blur", td->quad_sigma)
        IF("detector.refine", td->refine_edges)
        IF("detector.sharpening", td->decode_sharpening)
        IF("detector.debug", td->debug)
        IF("detector.clip_max", clip_max)
        IF("max_hamming", max_hamming)
        IF("profile", profile)
        IF("publish_tf", publish_tf)
        IF("size", tag_edge_size)
    }

    mutex.unlock();

    result.successful = true;

    return result;
}

void AprilTagNode::toggle_enabled(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response
)
{   if (request->data){
        RCLCPP_INFO(get_logger(), "Enabling apriltag detection");
    }
    else {
        RCLCPP_INFO(get_logger(), "Disabling apriltag detection");
    }
    scanning = request->data;
    response->success = true;
}

rclcpp::QoS AprilTagNode::getQoS()
{
    bool qos_param = this->declare_parameter<bool>("use_system_default_qos", false);

    if (qos_param) {
        RCLCPP_INFO(get_logger(), "Using SystemDefaultsQoS (simulation mode).");
        return rclcpp::SystemDefaultsQoS(); 
    } else {
        RCLCPP_INFO(get_logger(), "Using KeepLast QoS (robot mode).");
        return rclcpp::QoS{rclcpp::KeepLast(1)}.best_effort();
    }
}