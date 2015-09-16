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

abstract class Search {
  val name: String
  val results = new mutable.HashMap[ArmInfo, Array[Arm[_]]]()

  def appendResults(armInfo: ArmInfo, arms: Array[Arm[_]]) = {
    results(armInfo) = arms
  }

  def search(totalBudget: Int, arms: Map[(String, String), Arm[_]]): Arm[_]
}

class StaticSearch extends Search {
  override val name: String = "static search"
  override def search(totalBudget: Int, arms: Map[(String, String), Arm[_]]): Arm[_] = {

    assert(arms.keys.size != 0, "ERROR: No arms!")
    val armValues = arms.values.toArray
    val numArms = arms.keys.size
    var i = 0
    while (i  < totalBudget) {
      armValues(i % numArms).pullArm()
      i += 1
    }

    val bestArm = armValues.maxBy(arm => arm.getResults(true, Some("validation"))(1))
    bestArm
  }
}

class SimpleBanditSearch extends Search {
  override val name: String = "simple bandit search"
  override def search(totalBudget: Int, arms: Map[(String, String), Arm[_]]): Arm[_] = {
    val numArms = arms.size
    val alpha = 0.3
    val initialRounds = math.max(1, (alpha * totalBudget / numArms).toInt)

    val armValues = arms.values.toArray

    for (i <- 0 until initialRounds) {
      armValues.foreach(_.pullArm())
    }

    var currentBudget = initialRounds * numArms
    val numPreSelectedArms = math.max(1, (alpha * numArms).toInt)

    val preSelectedArms = armValues.sortBy(_.getResults(true, Some("validation"))(1))
      .reverse.dropRight(numArms - numPreSelectedArms)

    while (currentBudget < totalBudget) {
      preSelectedArms(currentBudget % numPreSelectedArms).pullArm()
      currentBudget += 1
    }

    val bestArm = preSelectedArms.maxBy(arm => arm.getResults(true, Some("validation"))(1))
    bestArm
  }
}

class ExponentialWeightsSearch extends Search {
  override val name: String = "exponential weight search"
  override def search(totalBudget: Int, arms: Map[(String, String), Arm[_]]): Arm[_] = {
    val numArms = arms.size
    val alpha = 0.3
    val initialRounds = math.max(1, (alpha * totalBudget / numArms).toInt)

    val armValues = arms.values.toArray

    for (i <- 0 until initialRounds) {
      armValues.foreach(_.pullArm())
    }

    var currentBudget = initialRounds * numArms
    val numGoodArms = math.max(1, (alpha * numArms).toInt)

    val remainedArms = armValues
      .sortBy(_.getResults(true, Some("validation"))(1)).reverse.dropRight(numArms - numGoodArms)

    while (currentBudget < totalBudget) {
      remainedArms(currentBudget % numGoodArms).pullArm()
      currentBudget += 1
    }

    val bestArm = remainedArms.maxBy(arm => arm.getResults(true, Some("validation"))(1))
    bestArm
  }
}

