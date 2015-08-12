/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.tuning.bandit

import scala.util.Random

/**
 * Sample parameters given a range of continuous or categorical number.
 */
abstract class ParamSampler[T](val name: String, val paramType: String) {

  /**
   * Sample an array of parameters.
   */
  def getSamples(numVal: Int): Array[T]

  /**
   * Sample one parameter.
   */
  def getOneRandomSample: T
}

class IntParamSampler(
    override val name: String,
    val minVal: Int,
    val maxVal: Int,
    val initVal: Int = None) extends ParamSampler[Int](name, _) {

  override val paramType: String = "integer"

  override def getSamples(numVal: Int): Array[Int] = {
    ParamSampler.evenlySamplingFromRange(minVal.toDouble, maxVal.toDouble, numVal).map(_.toInt)
  }

  override def getOneRandomSample: Int = {
    ParamSampler.samplingOneFromRange(minVal.toDouble, maxVal.toDouble).toInt
  }
}

/**
 * Sample Double parameters given a range of Double values.
 * @param scale default "log", which represents evenly sampling in a log space. Otherwise, the
 *              sampling is performed in the linear space.
 */
class DoubleParamSampler(
    override val name: String,
    val minVal: Double,
    val maxVal: Double,
    val scale: String = "log",
    val initVal: Double = None) extends ParamSampler[Double](name, _) {

  override val paramType: String = "continuous"

  override def getSamples(numVal: Int): Array[Double] = {
    if (this.scale == "log") {
      val minExponent = math.log10(minVal)
      val maxExponent = math.log10(maxVal)
      ParamSampler.evenlySamplingFromRange(minExponent, maxExponent, numVal)
        .map(x => math.pow(10, x))
    } else {
      ParamSampler.evenlySamplingFromRange(minVal, maxVal, numVal)
    }
  }

  override def getOneRandomSample: Double = {
    if (this.scale == "log") {
      val minExponent = math.log10(minVal)
      val maxExponent = math.log10(maxVal)
      math.pow(10, ParamSampler.samplingOneFromRange(minExponent, maxExponent))
    } else {
      ParamSampler.samplingOneFromRange(minVal, maxVal)
    }
  }
}

object ParamSampler {
  def evenlySamplingFromRange(minVal: Double, maxVal: Double, numVal: Int): Array[Double] = {
    val stepSize = (maxVal - minVal) * 1.0 / numVal
    val result = Array.fill[Double](numVal)(0)
    var i = 0
    while (i < numVal) {
      result(i) = minVal + i * stepSize
      i += 1
    }
    result
  }

  def samplingOneFromRange(minVal: Double, maxVal: Double): Double = {
    Random.nextDouble() * (1 + maxVal - minVal) + minVal
  }
}
