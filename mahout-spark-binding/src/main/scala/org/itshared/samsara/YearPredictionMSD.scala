package org.itshared.samsara

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.spark.SparkConf
import org.apache.mahout.sparkbindings._
import org.apache.spark.SparkContext
import collection._
import JavaConversions._
import org.apache.mahout.math.decompositions.DQR._
import org.apache.mahout.math.decompositions.DSPCA._
import org.apache.mahout.math.decompositions.DSSVD._
import scala.util.Random

object YearPredictionMSD {

  def main(args: Array[String]) {
    val textFilePath = "file:///c:/tmp/203/YearPredictionMSD.txt"
    val numPartitions = 2

    val conf = new SparkConf()
      .setAppName("Cereals OLS Regression")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "org.apache.mahout.sparkbindings.io.MahoutKryoRegistrator")
    implicit val sc = new SparkDistributedContext(new SparkContext(conf))

    val data = sc.textFile(textFilePath, numPartitions)
    val rdd = data.map(line => line.split(",").map(_.toDouble)).map(dvec(_))
    val rddMatrixLike: DrmRdd[Int] = rdd.zipWithIndex.map { case (v, idx) => (idx.toInt, v) }

    val matrix = drmWrap(rddMatrixLike)
    val nrow = matrix.nrow.toInt
    val ncol = matrix.ncol

    val train = matrix(0 until 463715, ::)
    val test = matrix(463715 until nrow, ::)

    val X = train(::, 1 until ncol)
    val y = train.viewColumn(0)

    val w = qrOls(addBias(X), y)

    val X_test = test(::, 1 until ncol)
    val y_test = test.viewColumn(0)

    val error = goodnessOfFit(addBias(X_test), w, y_test)
    println("error ols", error)

    val X_std = addBias(standardize(X))
    val X_test_std = addBias(standardize(X_test))
    val ridgeFunc = ridge(X_std, y)

    var best_lambda = 0.0
    var best_fit = Double.MaxValue

    for (lambda <- Seq(0.1, 0.5, 1.0, 3.0, 5.0, 10.0)) {
      val w_reg = ridgeFunc(lambda)
      val error_reg = goodnessOfFit(X_test_std, w_reg, y_test)
      println("error ridge", s"lambda = $lambda", error_reg)
      if (error_reg < best_fit) {
        best_fit = error_reg
        best_lambda = lambda
      }
    }

    println("best fit ridge", s"lambda = $best_lambda", best_fit)

    val X_subset = center(sample(X, 5000, seed = 0x31337))
    val pca = dimRed(X_subset, dim = 20)
    val w_pca = ols(addBias(pca(center(X))), y)
    val error_pca = goodnessOfFit(addBias(pca(center(X_test))), w_pca, y_test)
    println("error pca", error_pca)
  }

  def addBias(drmX: DrmLike[Int]) = {
    drmX.mapBlock(drmX.ncol + 1) {
      case (keys, block) => {
        val block_new = block.like(block.nrow, block.ncol + 1)
        block_new.zip(block).foreach {
          case (row_new, row_orig) =>
            row_new(0) = 1.0
            row_new(1 to block.ncol) := row_orig
        }
        keys -> block_new
      }
    }
  }

  def center(drmX: DrmLike[Int])(implicit ctx: DistributedContext) = {
    val mean = drmBroadcast(drmX.colMeans)

    val res = drmX.mapBlock(drmX.ncol) {
      case (keys, block) => {
        val copy = block.cloned
        copy.foreach(row => row := row - mean)
        (keys, copy)
      }
    }

    res
  }

  def center(mat: Matrix) = {
    val mean = mat.colMeans
    val mat_new = mat.cloned
    mat_new.foreach(row => row := row - mean)
    mat_new
  }

  def standardize(drmX: DrmLike[Int])(implicit ctx: DistributedContext) = {
    val meanVec = drmX.colMeans
    val variance = (drmX * drmX).colMeans - (meanVec * meanVec)

    val mean = drmBroadcast(meanVec)
    val std = drmBroadcast(variance.sqrt)

    val res = drmX.mapBlock(drmX.ncol) {
      case (keys, block) => {
        val copy = block.cloned
        copy.foreach(row => row := (row - mean) / std)
        (keys, copy)
      }
    }

    res
  }

  def ols(drmX: DrmLike[Int], y: Vector) = {
    val sol = solve(drmX.t %*% drmX, drmX.t %*% y)
    sol(::, 0)
  }

  def qrOls(drmX: DrmLike[Int], y: Vector) = {
    val QR = dqrThin(drmX)
    val Q = QR._1
    val R = QR._2

    val sol = solve(R, Q.t %*% y)
    sol(::, 0)
  }

  def ridge(drmX: DrmLike[Int], y: Vector): (Double) => Vector = {
    val XTX = drmX.t %*% drmX
    val XTy = drmX.t %*% y

    (lambda: Double) => {
      val reg = diag(lambda, XTX.ncol)
      val sol = solve(XTX.plus(reg), XTy)
      sol(::, 0)
    }
  }

  def dimRed(X_train: Matrix, dim: Int): DrmLike[Int] => DrmLike[Int] = {
    val V = svd(X_train)._2
    val V_red = V(::, 0 until dim)

    def fit(X: DrmLike[Int]): DrmLike[Int] = {
      X %*% V_red
    }

    fit
  }

  def goodnessOfFit(drmX: DrmLike[Int], w: Vector, y: Vector) = {
    val fittedY = (drmX %*% w) collect (::, 0)
    (y - fittedY).norm(2)
  }

  def sample(drm: DrmLike[Int], size: Int, seed: Long = System.currentTimeMillis) = {
    val nrow = drm.nrow.toInt
    val rnd = new Random(seed)

    val randomIdx = Seq.fill(size)(rnd.nextInt(nrow))
    val Srows = randomIdx.map { i =>
      val vec = new RandomAccessSparseVector(nrow)
      vec.setQuick(i, 1)
      vec
    }

    val S = new SparseRowMatrix(size, nrow)
    S := Srows
    S %*% drm
  }

}