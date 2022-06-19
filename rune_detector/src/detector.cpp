// Copyright (c) 2022 ChenJun
// Licensed under the MIT License.

#include "rune_detector/detector.hpp"

#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

namespace rm_power_rune
{
cv::Mat Detector::binarize(const cv::Mat & src)
{
  // Split the image into three channels
  std::vector<cv::Mat> channels;
  cv::split(src, channels);

  // Subtract between the red and blue channels
  cv::Mat diff;
  if (detect_color == Color::RED) {
    // 0-R 1-G 2-B
    cv::subtract(channels[0], channels[2], diff);
  } else {
    cv::subtract(channels[2], channels[0], diff);
  }

  // Threshold the image
  cv::threshold(diff, bin_, bin_thresh, 255, cv::THRESH_BINARY);

  return bin_;
}

cv::Mat Detector::floodfill()
{
  // Inverse the image
  cv::Mat inv;
  cv::bitwise_not(bin_, inv);

  cv::Mat tmp = inv.clone();
  // Floodfill the image from top left corner
  floodFill(inv, cv::Point(0, 0), cv::Scalar(0));
  floodfilled_ = inv;

  // If there is closed area at the top left corner, which causes the floodfill to fail,
  if (countNonZero(floodfilled_) > floodfilled_.rows * floodfilled_.cols * 0.5) {
    // Floodfill again from the bottom right corner
    floodFill(tmp, cv::Point(bin_.cols - 1, bin_.rows - 1), cv::Scalar(0));
    floodfilled_ = tmp;
  }

  return floodfilled_;
}

bool Detector::findArmor(cv::RotatedRect & armor)
{
  // Find the contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(floodfilled_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<cv::RotatedRect> armors;
  std::vector<cv::RotatedRect> strips;
  for (const auto & contour : contours) {
    // Approximate the contour
    auto rect = cv::minAreaRect(contour);

    // Use area to check if the contour is part of a rune
    if (rect.size.area() > min_armor_area) {
      // Use ratio to distinguish rect is armor or not
      float ratio = rect.size.width > rect.size.height ? rect.size.width / rect.size.height
                                                       : rect.size.height / rect.size.width;
      std::cout << "ratio: " << ratio << std::endl;

      if (min_armor_ratio < ratio && ratio < max_armor_ratio) {
        armors.emplace_back(rect);
      } else if (min_strip_ratio < ratio && ratio < max_strip_ratio) {
        strips.emplace_back(rect);
      }
    } else {
      continue;
    }
  }

  // Find available armor
  auto armor_it = armors.begin();
  while (armor_it != armors.end()) {
    bool available = true;

    auto strip_it = strips.begin();
    while (strip_it != strips.end()) {
      cv::Point2f armor_pts[4], strip_pts[4];
      armor_it->points(armor_pts);
      strip_it->points(strip_pts);

      // Calculate the min distance between the armor and the strip
      double min_dist = std::numeric_limits<double>::max();
      for (const auto & armor_pt : armor_pts) {
        for (const auto & strip_pt : strip_pts) {
          auto dist = cv::norm(armor_pt - strip_pt);
          min_dist = std::min(min_dist, dist);
        }
      }

      // Normalize the distance
      min_dist /= std::max(armor_it->size.width, armor_it->size.height);

      // The armor is close enough to the strip
      if (min_dist < 0.25) {
        available = false;
        strip_it = strips.erase(strip_it);
      } else {
        strip_it++;
      }
    }

    if (!available) {
      armor_it = armors.erase(armor_it);
    } else {
      armor_it++;
    }
  }

  if (armors.size() == 1) {
    this->armor_ = armors[0];
    armor = armors[0];
    return true;
  } else {
    return false;
  }
}

}  // namespace rm_power_rune
