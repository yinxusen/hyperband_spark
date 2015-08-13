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

import scala.collection.mutable

case class ArmInfo(dataName: String, numArms: Int, maxIter: Int, trial: Int)

abstract class SearchStrategy(
    val name: String,
    val allResults: mutable.Map[ArmInfo, Array[Arm[_]]]
      = new mutable.HashMap[ArmInfo, Array[Arm[_]]]()) {

  def appendResults(armInfo: ArmInfo, arms: Array[Arm[_]]) = {
    allResults(armInfo) = arms
  }

  def search(
      modelFamilies: Array[ModelFamily],
      maxIter: Int,
      arms: Map[(String, String), Arm[_]]): Arm[_]
}

class StaticSearchStrategy(
    override val name: String,
    override val allResults: mutable.Map[ArmInfo, Array[Arm[_]]])
  extends SearchStrategy(name, allResults) {

  override def search(
      modelFamilies: Array[ModelFamily],
      maxIter: Int,
      arms: Map[(String, String), Arm[_]]): Arm[_] = {

    assert(arms.keys.size != 0, "ERROR: No arms!")
    val armValues = arms.values.toArray
    val numArms = arms.keys.size
    var i = 0
    while (i  < maxIter) {
      armValues(i % numArms).pullArm()
      i += 1
    }

    val bestArm = armValues.minBy(arm => arm.getResults(true, Some("validation"))(1))
    bestArm
  }
}

