package org.apache.spark.ml.tuning.bandit

import org.scalatest.FunSuite

import scala.util.Random

class UtilsSuite extends FunSuite with BanditTestContext {
  test("Test dataset splitting") {
    val dataset = sc.parallelize(Array.range(0, 10))
    val (train, test) = Utils.splitTrainTest(dataset, 0.3, Random.nextInt())
    val localTrain = train.collect()
    val localTest = test.collect()
    for (x <- localTest) {
      assert(!localTrain.contains(x), "Element overlaps between training and test set.")
    }
    assert(localTest.size + localTrain.size == 10, "Element misses after splitting.")
  }
}
