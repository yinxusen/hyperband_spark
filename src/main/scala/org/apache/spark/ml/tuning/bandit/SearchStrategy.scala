package org.apache.spark.ml.tuning.bandit

import scala.collection.mutable

/**
 * Created by panda on 7/31/15.
 */

case class ArmInfo(dataName: String, numArms: Int, maxIter: Int, trial: Int)

abstract class SearchStrategy(
    val name: String,
    val allResults: mutable.Map[ArmInfo, Array[Arm[_]]] = Map.empty) {

  def appendResults(armInfo: ArmInfo, arms: Array[Arm[_]]) = {
    allResults(armInfo) = arms
  }

  def search(modelFamilies: Array[ModelFamily], maxIter: Int, arms: Map[(String, String), Arm]): Arm
}

class StaticSearchStrategy(
    override val name: String,
    override val allResults: mutable.Map[ArmInfo, Array[Arm[_]]])
  extends SearchStrategy(name, allResults) {

  override def search(
      modelFamilies: Array[ModelFamily],
      maxIter: Int,
      arms: Map[(String, String), Arm]): Arm = {

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

