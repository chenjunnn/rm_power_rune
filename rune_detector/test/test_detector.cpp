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
  detector->bin_thresh = 80;

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
  floodfilled = detector->floodfill();

  cv::imwrite("/tmp/floodfilled.png", floodfilled);
}

TEST(test_detector, findArmor)
{
  detector->min_armor_ratio = 1.5;
  detector->max_armor_ratio = 2.3;
  detector->min_strip_ratio = 3.8;
  detector->max_strip_ratio = 4.6;

  cv::RotatedRect armor;
  EXPECT_TRUE(detector->findArmor(armor));

  // Draw the armor
  cv::Mat src_with_armor = src.clone();
  cv::cvtColor(src_with_armor, src_with_armor, cv::COLOR_RGB2BGR);
  cv::ellipse(src_with_armor, armor, cv::Scalar(0, 255, 0), 2);

  cv::imwrite("/tmp/armor.png", src_with_armor);
}
