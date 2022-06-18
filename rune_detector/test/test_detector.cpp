// Copyright 2022 Chen Jun

#include <gtest/gtest.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/opencv.hpp>

// STL
#include <memory>

#include "rune_detector/detector.hpp"

std::unique_ptr<rm_power_rune::Detector> detector;

cv::Mat src;
cv::Mat bin;
cv::Mat floodfilled;

TEST(test_detector, init)
{
  detector = std::make_unique<rm_power_rune::Detector>();
  detector->detect_color = rm_power_rune::Detector::Color::RED;
  detector->thresh = 100;

  src =
    cv::imread(ament_index_cpp::get_package_share_directory("rune_detector") + "/sample/test.png");
  // BGR -> RGB
  cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
}

TEST(test_detector, binarize)
{
  bin = detector->binarize(src);

  cv::imwrite("/tmp/bin.png", bin);
}

TEST(test_detector, floodfill)
{
  floodfilled = detector->floodfill(bin);

  cv::imwrite("/tmp/floodfilled.png", floodfilled);
}
