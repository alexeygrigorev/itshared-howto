package org.itshared.samsara

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest._
import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.sparkbindings._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import collection._
import JavaConversions._

@RunWith(classOf[JUnitRunner])
class LATest extends FunSuite {

  test("add_bias") {
    val conf = new SparkConf()
      .setAppName("Cereals OLS Regression")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "org.apache.mahout.sparkbindings.io.MahoutKryoRegistrator")
    implicit val sc = new SparkDistributedContext(new SparkContext(conf))

    val matrix = dense(
      (2, 2, 10.5, 10, 29.509541),
      (1, 2, 12, 12, 18.042851),
      (1, 1, 12, 13, 22.736446),
      (3, 3, 13, 4, 45.811716))

    val expected = dense(
      (1, 2, 2, 10.5, 10, 29.509541),
      (1, 1, 2, 12, 12, 18.042851),
      (1, 1, 1, 12, 13, 22.736446),
      (1, 3, 3, 13, 4, 45.811716))

    val drmData = drmParallelize(matrix, numPartitions = 2)
    val actual = YearPredictionMSD.add_bias(drmData).collect

    assert((actual - expected).norm < 0.001)
  }

  test("standardize") {
    val conf = new SparkConf()
      .setAppName("Cereals OLS Regression")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "org.apache.mahout.sparkbindings.io.MahoutKryoRegistrator")
    implicit val sc = new SparkDistributedContext(new SparkContext(conf))

    System.setProperty("mahout.math.AewB.inplace", "false")

    val matrix = dense(
      (2, 2, 10.5, 10, 29.509541), // Apple Cinnamon Cheerios
      (1, 2, 12, 12, 18.042851), // Cap'n'Crunch
      (1, 1, 12, 13, 22.736446), // Cocoa Puffs
      (2, 1, 11, 13, 32.207582), // Froot Loops
      (1, 2, 12, 11, 21.871292), // Honey Graham Ohs
      (2, 1, 16, 8, 36.187559), // Wheaties Honey Gold
      (6, 2, 17, 1, 50.764999), // Cheerios
      (3, 2, 13, 7, 40.400208), // Clusters
      (3, 3, 13, 4, 45.811716)) // Great Grains Pecan)

    val expected = dense(
      (-0.2236068, 0.35355339, -1.18616051, 0.31038296, -0.33501565),
      (-0.89442719, 0.35355339, -0.45828929, 0.81828234, -1.41725863),
      (-0.89442719, -1.23743687, -0.45828929, 1.07223203, -0.97427027),
      (-0.2236068, -1.23743687, -0.94353677, 1.07223203, -0.0803706),
      (-0.89442719, 0.35355339, -0.45828929, 0.56433265, -1.05592477),
      (-0.2236068, -1.23743687, 1.48270063, -0.19751643, 0.29526545),
      (2.45967478, 0.35355339, 1.96794811, -1.97516427, 1.67110556),
      (0.4472136, 0.35355339, 0.02695819, -0.45146612, 0.69286143),
      (0.4472136, 1.94454365, 0.02695819, -1.21331519, 1.20360747))

    val drmData = drmParallelize(matrix, numPartitions = 2)
    val actual = YearPredictionMSD.standardize(drmData).collect

    assert((actual - expected).norm < 0.001)
  }

}