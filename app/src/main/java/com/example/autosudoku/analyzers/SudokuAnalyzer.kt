package com.example.autosudoku.analyzers

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.graphics.Rect
import android.media.Image
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.autosudoku.MainActivity
import com.example.autosudoku.solvers.Sudoku
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.ByteArrayOutputStream
import java.io.IOException
import kotlin.math.abs

/** Helper type alias used for analysis use case callbacks */
typealias BitmapListener = (bitmap: Bitmap) -> Unit

private const val TAG = "CannyEdge"

class CannyEdge(listener: BitmapListener? = null, context: Context) : ImageAnalysis.Analyzer {

    private val mModule: Module? = try {
        Log.d(TAG, "Pytorch - model loading")
        val module = LiteModuleLoader.load(
            MainActivity.assetFilePath(context,"cnn_with_typed_digits_diff_font_size_v2.ptl")
        )
        Log.d(TAG, "Pytorch - model loaded")
        module
    } catch (e: IOException) {
        Log.d(TAG, "Pytorch - model not found")
        null
    }

    private var mMats = ArrayList<Mat>().also{ for (i in 0 until 2) it.add(Mat(0, 0, 0)) }
    private var curMat = 0
    private val listeners = ArrayList<BitmapListener>().apply { listener?.let { add(it) } }
    private var frameCount = 0
    private val cellSize = 28
    private val bboxSize = cellSize * 9
    private val threshold = 60
    private val inTensorBuffer = Tensor.allocateFloatBuffer(cellSize * cellSize)

    private var lastMatrix: Array<IntArray>? = null

    private fun imgToBitmap(image: Image): Bitmap {
        val planes = image.planes
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer[nv21, 0, ySize]
        vBuffer[nv21, ySize, vSize]
        uBuffer[nv21, ySize + vSize, uSize]
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 75, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun FloatArray.maxElementIndex() = run {
        var index = 0
        var maxValue = Float.MIN_VALUE
        for (ind in 0 until size) {
            if (get(ind) > maxValue) {
                maxValue = get(ind)
                index = ind
            }
        }
        index
    }

    private fun filterDigits(image: Mat, cellSize: Int, bboxSize: Int): Mat {

        val usedMask = Array(image.width()) { IntArray(image.height()) { 0 } }
        val result = Mat.zeros(Size(1.0 * image.width(), 1.0 * image.height()), image.type())
        val image_32f = Mat()
        val result_32f = Mat()
        image.convertTo(image_32f, CvType.CV_32F)
        result.convertTo(result_32f, CvType.CV_32F)

        var totalPointsChecked = 0
        var totalSkipped = 0

        var foundDigest = 0
        for (i in 0 until 9) {
            for (j in 0 until 9) {
                val origX = i * cellSize + cellSize / 2
                val origY = j * cellSize + cellSize / 2

                var usedPoints = 0
                var curPoint = 0
                val points = ArrayList<Pair<Int, Int>>()
                for (xOffset in -5 until 6) {
                    for (yOffset in -5 until 6) {
                        val newX = origX + xOffset
                        val newY = origY + yOffset
                        usedMask[newX][newY] = 1
                        points.add(Pair(newX, newY))
                    }
                }

                while (curPoint < points.size) {
                    val (x, y) = points[curPoint]
                    curPoint += 1
                    totalPointsChecked += 1

                    val channels = FloatArray(1)
                    image_32f.get(x, y, channels)
                    if (channels[0] > threshold) {
                        result_32f.put(x, y, channels)
                        usedPoints += 1
                    }
                    for (xOffset in -2 until 2) {
                        for (yOffset in -2 until 2) {
                            val newX = x + xOffset
                            val newY = y + yOffset

                            if (newX >= bboxSize || newY >= bboxSize || newX < 0 || newY < 0) {
                                continue
                            }
                            if (abs(newX - x) > cellSize / 2 || abs(newY - y) > cellSize / 2) {
                                continue
                            }
                            if (usedMask[newX][newY] > 0) {
                                totalSkipped += 1
                                continue
                            }
                            if (image[newX, newY][0] > threshold) {
                                usedMask[newX][newY] = 1
                                points.add(Pair(newX, newY))
                            }
                        }
                    }
                }
                if (usedPoints < 10) {
                    for (point in points) {
                        result_32f.put(point.first, point.second, floatArrayOf(0.0f))
                    }
                } else {
                    foundDigest += 1
                }
            }
        }
        Log.d(TAG, "Found digest - $foundDigest, checked - $totalPointsChecked, skipped - $totalSkipped")
        result_32f.convertTo(result, CvType.CV_8UC1)
        return result
    }

    private fun preprocessOpenCV(bitmap: Bitmap): Bitmap? {
        val startTime = SystemClock.elapsedRealtime()

        val rgba = Mat()
        val gray = Mat()
        Utils.bitmapToMat(bitmap, mMats[curMat])

        Imgproc.GaussianBlur(mMats[curMat], mMats[curMat xor 1], Size(5.0, 5.0), 0.0)
        curMat = curMat xor 1
        Imgproc.medianBlur(mMats[curMat], mMats[curMat xor 1], 3)
        curMat = curMat xor 1

        mMats[curMat].copyTo(rgba)

        Imgproc.cvtColor(mMats[curMat], mMats[curMat xor 1], Imgproc.COLOR_RGBA2GRAY)
        curMat = curMat xor 1

        mMats[curMat].copyTo(gray)

        Imgproc.Canny(mMats[curMat], mMats[curMat xor 1], 60.0, 200.0)
        curMat = curMat xor 1
        Imgproc.threshold(mMats[curMat], mMats[curMat xor 1], 127.0, 255.0, 0)
        curMat = curMat xor 1

        val dilateKernel = Mat.ones(Size(3.0, 3.0), 0)
        val erodeKernel = Mat.ones(Size(5.0, 5.0), 0)
        Imgproc.dilate(mMats[curMat], mMats[curMat xor 1], dilateKernel, Point(-1.0, -1.0), 5)
        curMat = curMat xor 1
        Imgproc.erode(mMats[curMat], mMats[curMat xor 1], erodeKernel, Point(-1.0, -1.0), 1)
        curMat = curMat xor 1


        val contours = ArrayList<MatOfPoint>(0)
        Imgproc.findContours(mMats[curMat], contours, mMats[curMat xor 1], Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)
        contours.sortBy { -Imgproc.contourArea(it) }

        if (contours.size == 0) {
            return null
        }

        val hullIdx = MatOfInt()
        Imgproc.convexHull(contours[0], hullIdx)
        val contourArray = contours[0].toArray()
        val hullPoints = ArrayList<Point>()
        for (idx : Int in 0 until hullIdx.rows()) {
            hullPoints.add(contourArray[hullIdx.toList()[idx]])
        }
        val hull_2f = MatOfPoint2f()
        val bbox_2f = MatOfPoint2f()
        hull_2f.fromList(hullPoints)
        val contour_2f = MatOfPoint2f()
        contours[0].convertTo(contour_2f, CvType.CV_32FC2)

        val epsilon = 0.01 * Imgproc.arcLength(contour_2f, true)

        val bbox = MatOfPoint()
        val hull = MatOfPoint()
        Imgproc.approxPolyDP(hull_2f, bbox_2f, epsilon, true)
        bbox_2f.convertTo(bbox, CvType.CV_32S)
        hull_2f.convertTo(hull, CvType.CV_32S)

        /*Log.d(TAG, "contours.length - ${contours.size}")*/

        val hullArea = Imgproc.contourArea(hull)
        Log.d(TAG, "hull.size - $hullArea")
        if (hullArea < 30000) {
            return null
        }

        val newBbox = matchCorners(
            listOf(
                Point(0.0, 0.0), Point(bboxSize.toDouble(), 0.0),
                Point(0.0, bboxSize.toDouble()), Point(bboxSize.toDouble(), bboxSize.toDouble())
            )
        )
        val prevBbox = matchCorners(bbox.toList())
        val transform = Imgproc.getPerspectiveTransform(prevBbox, newBbox)

        Imgproc.warpPerspective(gray, mMats[curMat], transform, Size(bboxSize.toDouble(), bboxSize.toDouble()))
        Imgproc.warpPerspective(rgba, rgba, transform, Size(bboxSize.toDouble(), bboxSize.toDouble()))

        Imgproc.adaptiveThreshold(mMats[curMat], mMats[curMat xor 1], 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 3, 5.0)
        curMat = curMat xor 1

        Imgproc.GaussianBlur(mMats[curMat], mMats[curMat xor 1], Size(3.0, 3.0), 0.0)
        curMat = curMat xor 1

        mMats[curMat xor 1] = filterDigits(mMats[curMat], cellSize, bboxSize)
        curMat = curMat xor 1

        Imgproc.cvtColor(mMats[curMat], rgba, Imgproc.COLOR_GRAY2RGBA)

        val buffer = ByteArray(cellSize * cellSize)
        val torchStartTime = SystemClock.elapsedRealtime()


        val sudokuMatrix = Array(9) { IntArray(9) { 0 } }

        for (i in 0 until 9) {
            for (j in 0 until 9) {
                mMats[curMat].submat(j * cellSize, (j + 1) * cellSize, i * cellSize, (i + 1) * cellSize).get(0, 0, buffer)

                val floatBuffer = buffer.map{ it.toUByte().toFloat() / 255}.toFloatArray()
                var nonNullElements = 0
                floatBuffer.forEach {
                    if (it > 0.5) {
                        nonNullElements += 1
                    }
                }
                if (nonNullElements < 15) {
                    continue
                }
                inTensorBuffer.put(floatBuffer)
                val inputTensor = Tensor.fromBlob(
                    inTensorBuffer,
                    longArrayOf(1, 1, cellSize.toLong(), cellSize.toLong())
                )
                inTensorBuffer.position(0)
                val outputTensor = mModule!!.forward(IValue.from(inputTensor)).toTensor()
                val scores = outputTensor.dataAsFloatArray
                sudokuMatrix[i][j] = scores.maxElementIndex()
                if (sudokuMatrix[i][j] > 9) {
                    sudokuMatrix[i][j] = 0
                }
            }
        }
        val endTime = SystemClock.elapsedRealtime()
        Log.d(TAG, "Pytorch inference time - ${endTime - torchStartTime}")
        Log.d(TAG, "Total inference time - ${endTime - startTime}")

        val solution = Sudoku().solve(sudokuMatrix)

        if (solution.isSuccess || lastMatrix == null) {
            lastMatrix = solution.matrix
        }

        /*Log.d(TAG, "Has solution - ${solution.isSuccess}")*/
        for (i in 0 until 9) {
            for (j in 0 until 9) {

                val color = when {
                    sudokuMatrix[i][j] != 0 -> Scalar(0.0, 255.0, 0.0)
                    else -> Scalar(255.0, 0.0)
                }

                Imgproc.putText(
                    rgba, lastMatrix!![i][j].toString(), Point(1.0 * cellSize * i + cellSize / 2, 1.0 * cellSize * j + cellSize / 2),
                    Imgproc.FONT_ITALIC, 0.7, color
                )
            }
        }
        val newBitmap = Bitmap.createBitmap(mMats[curMat].width(), mMats[curMat].height(), Bitmap.Config.ARGB_8888) /*bitmap.copy(Bitmap.Config.ARGB_8888, true)*/
        Utils.matToBitmap(rgba, newBitmap)
        return newBitmap
    }

    private fun innerAnalyze(bitmap: Bitmap) {
        val newBitmap = preprocessOpenCV(bitmap) ?: return
        listeners.forEach { it(newBitmap) }
    }

    @SuppressLint("UnsafeOptInUsageError")
    override fun analyze(image: ImageProxy) {
        if (image.image == null) {
            return
        }
        frameCount += 1
        if (frameCount % 10 != 0) {
            image.close()
            return
        }

        val matrix = Matrix()
        matrix.postRotate(90.0f)
        var bitmap = imgToBitmap(image.image!!)
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        mMats[curMat] = Mat.zeros(bitmap.width, bitmap.height, 0)
        mMats[curMat xor 1] = Mat.zeros(bitmap.width, bitmap.height, 0)

        innerAnalyze(bitmap)

        mMats[curMat].release()
        mMats[curMat xor 1].release()
        image.close()
    }

    private fun matchCorners(points: List<Point>): MatOfPoint2f {
        fun matchCorner(points: List<Point>, align: Point): Point {
            return points.sortedBy { -it.dot(align) }[0]
        }
        val aligns = listOf(Point(1.0, -1.0), Point(1.0, 1.0), Point(-1.0, 1.0), Point(-1.0, -1.0))
        val res = MatOfPoint2f()
        res.fromList(aligns.map{ matchCorner(points, it) })
        return res
    }
}
