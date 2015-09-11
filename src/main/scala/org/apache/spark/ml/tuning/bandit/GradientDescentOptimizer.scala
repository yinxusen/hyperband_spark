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

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{DenseVector => BDV, norm}

import org.apache.spark.annotation.{Experimental, DeveloperApi}
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.optimization.{Gradient, Updater, Optimizer}


/**
 * Class used to solve an optimization problem using Gradient Descent.
 * @param gradient Gradient function to be used.
 * @param updater Updater to be used to update weights after every iteration.
 */
class GradientDescentOptimizer (private var gradient: Gradient, private var updater: Updater) extends Optimizer with Logging {

  private var stepSize: Double = 1.0
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  private var convergenceTol: Double = 0.001
  private var currentStep: Int = 1

  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }

  /**
   * :: Experimental ::
   * Set fraction of data to be used for each SGD iteration.
   * Default 1.0 (corresponding to deterministic/classical gradient descent)
   */
  @Experimental
  def setMiniBatchFraction(fraction: Double): this.type = {
    this.miniBatchFraction = fraction
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the convergence tolerance. Default 0.001
   * convergenceTol is a condition which decides iteration termination.
   * The end of iteration is decided based on below logic.
   * - If the norm of the new solution vector is >1, the diff of solution vectors
   *   is compared to relative tolerance which means normalizing by the norm of
   *   the new solution vector.
   * - If the norm of the new solution vector is <=1, the diff of solution vectors
   *   is compared to absolute tolerance which is not normalizing.
   * Must be between 0.0 and 1.0 inclusively.
   */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(0.0 <= tolerance && tolerance <= 1.0)
    this.convergenceTol = tolerance
    this
  }

  /**
   * Set the gradient function (of the loss function of one single data example)
   * to be used for SGD.
   */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
   * Set the updater function to actually perform a gradient step in a given direction.
   * The updater is responsible to perform the update from the regularization term as well,
   * and therefore determines what kind or regularization is used, if any.
   */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  def setCurrentStep(cs: Int): this.type = {
    this.currentStep = cs
    this
  }

  /**
   * :: DeveloperApi ::
   * Runs gradient descent on the given training data.
   * @param data training data
   * @param initialWeights initial weights
   * @return solution vector
   */
  @DeveloperApi
  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = GradientDescentOptimizer.runSingleStepSGD(
      data,
      gradient,
      updater,
      stepSize,
      currentStep,
      regParam,
      miniBatchFraction,
      initialWeights,
      convergenceTol)
    weights
  }

}

/**
 * :: DeveloperApi ::
 * Top-level method to run gradient descent.
 */
@DeveloperApi
object GradientDescentOptimizer extends Logging {
  def runSingleStepSGD(
      data: RDD[(Double, Vector)],
      gradient: Gradient,
      updater: Updater,
      stepSize: Double,
      currentStep: Int,
      regParam: Double,
      miniBatchFraction: Double,
      initialWeights: Vector,
      convergenceTol: Double): Vector = {

    // Initialize weights as a column vector
    val weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size

    val bcWeights = data.context.broadcast(weights)
    val (gradientSum, lossSum, miniBatchSize) = data.sample(false, miniBatchFraction, 42 + currentStep)
      .treeAggregate((BDV.zeros[Double](n), 0.0, 0L))(
        seqOp = (c, v) => {
          // c: (grad, loss, count), v: (label, features)
          val l = gradient.compute(v._2, v._1, bcWeights.value, Vectors.fromBreeze(c._1))
          (c._1, c._2 + l, c._3 + 1)
        },
        combOp = (c1, c2) => {
          // c: (grad, loss, count)
          (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
        })

    val update = updater.compute(
      weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble),
      stepSize, currentStep, regParam)

    update._1
  }
}
