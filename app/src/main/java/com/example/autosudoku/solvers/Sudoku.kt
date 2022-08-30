package com.example.autosudoku.solvers

import android.util.Log

private const val TAG = "SudokuSolver"

class Sudoku {

    data class Solution(val isSuccess: Boolean, val matrix: Array<IntArray>)

    private var mSteps = 0

    private fun setElement(matrix: Array<IntArray>, possibilities: Array<IntArray>, row: Int, col: Int, value: Int) {
        val bitValue = 1 shl value
        for (i in 0 until 9) {
            val newRow = (row / 3) * 3 + i / 3
            val newCol = (col / 3) * 3 + i % 3
            possibilities[i][col] = possibilities[i][col] or bitValue
            possibilities[row][i] = possibilities[row][i] or bitValue
            possibilities[newRow][newCol] = possibilities[newRow][newCol] or bitValue
        }
        matrix[row][col] = value
    }

    private fun minZeroBitIndex(value: Int): Int {
        for (ind in 1 until 10) {
            if ((value and (1 shl ind)) == 0) {
                return ind
            }
        }
        return -1
    }

    private fun Array<IntArray>.copy() = Array(size) { get(it).clone() }

    private fun recSolve(matrix: Array<IntArray>, possibilities: Array<IntArray>): Solution {
        mSteps += 1
        var maxCell = Triple(0, 0, 0)
        var skipped = 0
        for (i in 0 until 9) {
            for (j in 0 until 9) {
                if (matrix[i][j] != 0) {
                    skipped += 1
                    continue
                }
                val bits = possibilities[i][j].countOneBits()
                if (bits >= maxCell.first) {
                    maxCell = Triple(bits, i, j)
                }
            }
        }
        val (bits, row, col) = maxCell

        when {
            mSteps > 5000 -> {
                return Solution(false, matrix)
            }
            skipped == 9 * 9 -> {
                return Solution(true, matrix)
            }
            bits == 8 -> {
                setElement(matrix, possibilities, row, col, minZeroBitIndex(possibilities[row][col]))
                return recSolve(matrix, possibilities)
            }
            bits < 8 -> {
                for (bitInd in 1..9) {
                    if ((possibilities[row][col] and (1 shl bitInd)) == 0) {
                        val newMatrix = matrix.copy()
                        val newPossibilities = possibilities.copy()
                        setElement(newMatrix, newPossibilities, row, col, bitInd)
                        val solution = recSolve(newMatrix, newPossibilities)
                        if (solution.isSuccess) {
                            return solution
                        }
                    }
                }
            }
        }
        return Solution(false, matrix)
    }

    fun solve(originalMatrix: Array<IntArray>): Solution {
        mSteps = 0
        val matrix = Array(9) { IntArray(9) { 0 } }
        val possibilities = Array(9) { IntArray(9) { 0 } }
        for (i in 0 until 9) {
            for (j in 0 until 9) {
                if (originalMatrix[i][j] != 0) {
                    setElement(matrix, possibilities, i, j, originalMatrix[i][j])
                }
            }
        }
        Log.d(TAG, "Original")
        for (i in 0 until 9) {
            Log.d(TAG, matrix[i].joinToString(" "))
        }
        Log.d(TAG, " ")
        val solution = recSolve(matrix, possibilities)
        Log.d(TAG, "IsSolved - ${solution.isSuccess}, steps - $mSteps")
        Log.d(TAG, "Solution")
        for (i in 0 until 9) {
            Log.d(TAG, solution.matrix[i].joinToString(" "))
        }
        Log.d(TAG, " ")
        return solution
    }
}
