package org.apache.spark.ml.tuning.bandit

import org.apache.spark.mllib.linalg.{BLAS, Vectors}
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

  test("Randomly choose one") {
    val dist = (0 until 100).toArray.map(_ => Random.nextDouble())
    val selected = sc.parallelize(0 until 1000000).map(_ => Utils.chooseOne(dist))



    val selectedHistogram = selected.map(x => (x, 1)).reduceByKey(_ + _).collect()
      .sortBy(_._2).reverse
    val distWithIndex = dist.zipWithIndex.map(x => (x._2, x._1)).sortBy(_._2).reverse
      .dropRight(dist.size - selectedHistogram.size)

    for (key <- selectedHistogram.map(_._1)) {
      assert(key < 100, s"Selected element $key beyond scope.")
      assert(key >= 0, s"Selected element $key beyond scope.")
    }

    val distVector = Vectors.dense(distWithIndex.map(_._1.toDouble))
    val selectedVector = Vectors.dense(selectedHistogram.map(_._1.toDouble))

    BLAS.axpy(-1, distVector, selectedVector)

    val currentMissMatch = math.sqrt(selectedVector.toArray.map(x => x * x).sum)

    val totalMissMatch = math.sqrt((0 until 100).reverse.toArray.zipWithIndex.map(x => x._1 - x._2)
      .map(x => x * x).sum)

    assert(currentMissMatch / totalMissMatch < 0.5,
      "Selected elements miss match with original array.")

  }
}
