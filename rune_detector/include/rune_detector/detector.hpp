// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef RUNE_DETECTOR__DETECTOR_HPP_
#define RUNE_DETECTOR__DETECTOR_HPP_

// OpenCV
#include <opencv2/opencv.hpp>

// STD
#include <cmath>
#include <string>
#include <vector>

namespace rm_power_rune
{
class Detector
{
public:
  cv::Mat binarize(const cv::Mat & src);

  cv::Mat floodfill(const cv::Mat & bin);

  enum class Color {
    RED,
    BLUE,
  } detect_color;

  int thresh;

private:
};

}  // namespace rm_power_rune

#endif  // RUNE_DETECTOR__DETECTOR_HPP_
