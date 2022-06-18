// Copyright (c) 2022 ChenJun
// Licensed under the MIT License.

#include "rune_detector/detector.hpp"

#include <opencv2/core/mat.hpp>

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
  cv::Mat thresholded;
  cv::threshold(diff, thresholded, thresh, 255, cv::THRESH_BINARY);

  return thresholded;
}

cv::Mat Detector::floodfill(const cv::Mat & bin)
{
  // Inverse the image
  cv::Mat inv;
  cv::bitwise_not(bin, inv);

  cv::Mat floodfilled = inv.clone();
  cv::Mat tmp = inv.clone();
  // Floodfill the image from top left corner
  floodFill(floodfilled, cv::Point(0, 0), cv::Scalar(0));

  // If there is closed area at the top left corner, which causes the floodfill to fail,
  if (countNonZero(floodfilled) > floodfilled.rows * floodfilled.cols * 0.5) {
    // Floodfill again from the bottom right corner
    floodFill(tmp, cv::Point(bin.cols - 1, bin.rows - 1), cv::Scalar(0));
    floodfilled = tmp;
  }

  return floodfilled;
}

}  // namespace rm_power_rune
