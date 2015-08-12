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

import com.github.fommil.netlib.F2jBLAS
import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.{IntParam, Param, ParamMap, Params, _}
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, SQLContext}

/**
 * Params for [[BanditValidator]] and [[BanditValidatorModel]].
 */
trait BanditValidatorParams extends Params with HasStepsPerPulling with HasSeed {

  val problemType: Param[String] = new Param(this, "problemType", "types of problems")
  setDefault(problemType, "CLASSIFY")

  def getProblemType: String = $(problemType)

  val computeHistory: BooleanParam = new BooleanParam(this, "computeHistory",
    "whether to compute history or not")
  setDefault(computeHistory, true)

  def getComputeHistory: Boolean = $(computeHistory)

  val baselines: Param[Map[String, Double]] = new Param(this, "baselines",
    "baseline of each dataset")

  def getBaselines: Map[String, Double] = $(baselines)

  val modelFamilies: Param[Array[ModelFamily]] = new Param(this, "modelFamilies", "model families")

  def getModelFamilies: Array[ModelFamily] = $(modelFamilies)

  val numTrails: IntParam = new IntParam(this, "numTrails", "number of trails")

  def getNumTrails: Int = $(numTrails)

  val datasets: Param[Map[String, String]] = new Param(this, "datasets", "datasets to use")

  def getDatasets: Map[String, String] = $(datasets)

  val numArmsList: Param[Array[Int]] = new Param(this, "numArmsList",
    "a list of numbers of arms per parameter")

  def getNumArmsList: Array[Int] = $(numArmsList)

  val expectedIters: Param[Array[Int]] = new Param(this, "expectedIters", "expected iterations")

  def getExpectedIters: Array[Int] = $(expectedIters)

  val searchStrategies: Param[Array[SearchStrategy]] = new Param(this, "searchStrategies", "")

  def getSearchStrategies: Array[SearchStrategy] = $(searchStrategies)

  val evaluator: Param[Evaluator] = new Param(this, "evaluator",
    "evaluator used to select hyper-parameters that maximize the cross-validated metric")

  /** @group getParam */
  def getEvaluator: Evaluator = $(evaluator)
}

/**
 * :: Experimental ::
 * K-fold cross validation.
 */
@Experimental
class BanditValidator(override val uid: String)
  extends Estimator[BanditValidatorModel] with BanditValidatorParams with Logging {

  def this() = this(Identifiable.randomUID("bandit validation"))

  def transformSchema(schema: StructType): StructType = {
    schema
  }

  def copy(extra: ParamMap): BanditValidator = ???

  private val f2jBLAS = new F2jBLAS

  def setProblemType(value: String): this.type = set(problemType, value)

  def setComputeHistory(value: Boolean): this.type = set(computeHistory, value)

  def setBaselines(value: Map[String, Double]): this.type = set(baselines, value)

  def setModelFamilies(value: Array[ModelFamily]): this.type = set(modelFamilies, value)

  def setNumTrails(value: Int): this.type = set(numTrails, value)

  def setDatasets(value: Map[String, String]): this.type = set(datasets, value)

  def setNumArmsList(value: Array[Int]): this.type = set(numArmsList, value)

  def setExpectedIters(value: Array[Int]): this.type = set(expectedIters, value)

  def setSearchStrategies(value: Array[SearchStrategy]): this.type = set(searchStrategies, value)

  def setEvaluator(value: Evaluator): this.type = set(evaluator, value)

  def setStepsPerPulling(value: Int): this.type = set(stepsPerPulling, value)

  def setSeed(value: Long): this.type = set(seed, value)

  override def fit(dataset: DataFrame): BanditValidatorModel = ???

  def fit(sqlCtx: SQLContext) = {
    val results = $(datasets).flatMap { case (dataName, fileName) =>
      val data = ClassifyDataset.scaleAndPartitionData(sqlCtx, dataName, fileName)
      val allArms = Arms.generateArms($(modelFamilies), data, $(numArmsList).max).mapValues { arm =>
        arm.abridgedHistory.compute = $(computeHistory)
        arm
      }

      val armsAllocator = new ArmsAllocator(allArms)

      if ($(computeHistory)) {
        for ((armInfo, arm) <- allArms) {
          val maxIter = math.pow(2, 14)
          arm.trainToCompletion(maxIter)
          println(armInfo)
          println(arm.abridgedHistory.iterations.mkString(", "))
          println(arm.abridgedHistory.errors.mkString(", "))
        }
      }

      $(numArmsList).flatMap { case numArmsPerParameter =>
        val numArms = $(modelFamilies)
          .map(modelFamily => math.pow(numArmsPerParameter, modelFamily.paramList.size)).sum.toInt
        $(expectedIters).zipWithIndex.flatMap { case (expectedNumItersPerArm, idx) =>
          $(searchStrategies).map { case searchStrategy =>
            val arms = armsAllocator.allocate(numArms)
            val bestArm = searchStrategy
              .search($(modelFamilies), expectedNumItersPerArm * numArms, arms)
            ((searchStrategy.name, dataName, numArms, expectedNumItersPerArm),
              bestArm.getResults(false, None))
          }
        }
      }
    }
    results
  }
}

class BanditValidatorModel private[ml] (
    override val uid: String,
    val bestModel: Model[_])
  extends Model[BanditValidatorModel] with BanditValidatorParams {

  override def validateParams(): Unit = {
    bestModel.validateParams()
  }

  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    bestModel.transform(dataset)
  }

  override def transformSchema(schema: StructType): StructType = {
    bestModel.transformSchema(schema)
  }

  override def copy(extra: ParamMap): BanditValidatorModel = {
    val copied = new BanditValidatorModel(uid, bestModel.copy(extra).asInstanceOf[Model[_]])
    copyValues(copied, extra)
  }
}

