package com.example.autosudoku

import com.example.autosudoku.solvers.Sudoku
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
class ExampleUnitTest {

    @Test
    fun sudokuEmptyMatrixCorrect() {
        val matrix = Array(9) {IntArray(9) {0} }
        val solution = Sudoku().solve(matrix)
        assertEquals(solution.isSuccess, true)


        for (i in 0 until 9) {
            println(solution.matrix[i].joinToString(" "))
        }
    }
}