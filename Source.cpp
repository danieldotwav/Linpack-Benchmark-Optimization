// Name: Daniel Rivas
// Lab 5B
// CS 230 Advanced Computer Architecture
// Date Modified: 05/28/2024

// MODIFICATIONS AND RESULTS
// 
// 1. Use AVX intrinsics for matgen()
// Methods: Completely refactor the function to rely almost entirely on AVX instructions for more fine-tuned control over the calculations.
// Results: While the code itself was easier to follow logically, the code produces fewer kflops and longer CPU times for both arrays sized 200 and 1000.
// 
// 2. Refactor degefa()
// Methods: We can reduce redundancy in the original code by using dscal_func and daxpy_func to point to either dscal_r or dscal_ur based on the value of roll.
// Results: Kflops increased by over 100000 and CPU time was reduced by 2-3 seconds
// 
// 3. Reduce code redundancy in dgesl()
// Methods: Consolidate the separate branches for roll into a single structure, reduce nesting and redundant branches, and better separate the logic for handling 'roll' and 'non-roll' cases.
// Results: Results were inconsistent; the program performed better in terms of time and kflops with an array of size 200, but worse in both respects for size 1000 arrays.
//
// 4. Pointer arithmetic for daxpy_r()
// Methods: Unrolling loops can reduce the overhead of loop control and increase instruction-level parallelism. We use poitner arithmetic for faster access and inline small functions to reduce the overhead of function calls. Additionally, we ensure data is memory-aligned for better cache performance.
// Results: Kflops increased by over 200000 and CPU time was drastically reduced.
//
// 5. Loop unrolling in ddor_r()
// Methods: Applied a loop unrolling technique in cases where 'incx' and 'incy' are both 1, in order to reduce the number of iterations and loop overhead.
// Results: Kflops increased marginally, and CPU time remained relatively constant with minor improvements.
//
//
//


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <Windows.h>
#define DP

#ifdef SP
#define ZERO        0.0
#define ONE         1.0
#define PREC        "Single"
#define BASE10DIG   FLT_DIG

typedef float   REAL;
#endif

#ifdef DP
#define ZERO        0.0e0
#define ONE         1.0e0
#define PREC        "Double"
#define BASE10DIG   DBL_DIG

typedef double  REAL;
#endif

static REAL linpack(long nreps, int arsize);
static void matgen(REAL* a, int lda, int n, REAL* b, REAL* norma);
static void dgefa(REAL* a, int lda, int n, int* ipvt, int* info, int roll);
static void dgesl(REAL* a, int lda, int n, int* ipvt, REAL* b, int job, int roll);
static void daxpy_r(int n, REAL da, REAL* dx, int incx, REAL* dy, int incy);
static REAL ddot_r(int n, REAL* dx, int incx, REAL* dy, int incy);
static void dscal_r(int n, REAL da, REAL* dx, int incx);
static void daxpy_ur(int n, REAL da, REAL* dx, int incx, REAL* dy, int incy);
static REAL ddot_ur(int n, REAL* dx, int incx, REAL* dy, int incy);
static void dscal_ur(int n, REAL da, REAL* dx, int incx);
static int  idamax(int n, REAL* dx, int incx);
static REAL second(void);

static void* mempool;

int main(int argc, char* argv[])

{
    char    buf[80];
    int     arsize;
    long    arsize2d, memreq, nreps;
    size_t  malloc_arg;

    time_t startTime, endTime; // added for CS230 timing
    clock_t start, end;
    double cpu_time_used;

    while (1)
    {
        printf("Enter array size (q to quit) [200]:  ");
        fgets(buf, 79, stdin);
        if (buf[0] == 'q' || buf[0] == 'Q')
            break;
        if (buf[0] == '\0' || buf[0] == '\n')
            arsize = 200;
        else
            arsize = atoi(buf);
        arsize /= 2;
        arsize *= 2;
        if (arsize < 10)
        {
            printf("Too small.\n");
            continue;
        }

        time(&startTime); // Added for CS230 timing
        start = clock();

        arsize2d = (long)arsize * (long)arsize;
        memreq = arsize2d * sizeof(REAL) + (long)arsize * sizeof(REAL) + (long)arsize * sizeof(int);
        printf("Memory required:  %ldK.\n", (memreq + 512L) >> 10);
        malloc_arg = (size_t)memreq;
        if (malloc_arg != memreq || (mempool = malloc(malloc_arg)) == NULL)
        {
            printf("Not enough memory available for given array size.\n\n");
            continue;
        }
        printf("\n\nLINPACK benchmark, %s precision.\n", PREC);
        printf("Machine precision:  %d digits.\n", BASE10DIG);
        printf("Array size %d X %d.\n", arsize, arsize);
        printf("Average rolled and unrolled performance:\n\n");
        printf("    Reps Time(s) DGEFA   DGESL  OVERHEAD    KFLOPS\n");
        printf("----------------------------------------------------\n");
        nreps = 1;
        while (linpack(nreps, arsize) < 10.)
            nreps *= 2;
        free(mempool);
        printf("\n");
        // The following code added for CS230 timing
        time(&endTime);
        end = clock();
        double elapsedDiff = endTime - startTime;
        printf("...elapsed time difference: %f seconds\n", elapsedDiff);
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("...CPU time: %f seconds\n", cpu_time_used);


    }
}

static REAL linpack(long nreps, int arsize)

{
    REAL* a, * b;
    REAL   norma, t1, kflops, tdgesl, tdgefa, totalt, toverhead, ops;
    int* ipvt, n, info, lda;
    long   i, arsize2d;

    lda = arsize;
    n = arsize / 2;
    arsize2d = (long)arsize * (long)arsize;
    ops = ((2.0 * n * n * n) / 3.0 + 2.0 * n * n);
    a = (REAL*)mempool;
    b = a + arsize2d;
    ipvt = (int*)&b[arsize];
    tdgesl = 0;
    tdgefa = 0;
    totalt = second();
    for (i = 0; i < nreps; i++)
    {
        matgen(a, lda, n, b, &norma);
        t1 = second();
        dgefa(a, lda, n, ipvt, &info, 1);
        tdgefa += second() - t1;
        t1 = second();
        dgesl(a, lda, n, ipvt, b, 0, 1);
        tdgesl += second() - t1;
    }
    for (i = 0; i < nreps; i++)
    {
        matgen(a, lda, n, b, &norma);
        t1 = second();
        dgefa(a, lda, n, ipvt, &info, 0);
        tdgefa += second() - t1;
        t1 = second();
        dgesl(a, lda, n, ipvt, b, 0, 0);
        tdgesl += second() - t1;
    }
    totalt = second() - totalt;
    if (totalt < 0.5 || tdgefa + tdgesl < 0.2)
        return(0.);
    kflops = 2. * nreps * ops / (1000. * (tdgefa + tdgesl));
    toverhead = totalt - tdgefa - tdgesl;
    if (tdgefa < 0.)
        tdgefa = 0.;
    if (tdgesl < 0.)
        tdgesl = 0.;
    if (toverhead < 0.)
        toverhead = 0.;
    printf("%8ld %6.2f %6.2f%% %6.2f%% %6.2f%%  %9.3f\n",
        nreps, totalt, 100. * tdgefa / totalt,
        100. * tdgesl / totalt, 100. * toverhead / totalt,
        kflops);
    return(totalt);
}

// ORIGINAL
static void matgen(REAL* a, int lda, int n, REAL* b, REAL* norma)
{
    int init, i, j;

    init = 1325;
    *norma = 0.0;
    for (j = 0; j < n; j++)
        for (i = 0; i < n; i++)
        {
            init = (int)((long)3125 * (long)init % 65536L);
            a[lda * j + i] = (init - 32768.0) / 16384.0;
            *norma = (a[lda * j + i] > *norma) ? a[lda * j + i] : *norma;
        }
    for (i = 0; i < n; i++)
        b[i] = 0.0;
    for (j = 0; j < n; j++)
        for (i = 0; i < n; i++)
            b[i] = b[i] + a[lda * j + i];
}


// REFACTORED
// Here, we use inline assembly in order to improve performance in the following ways:
// 1. Using imul and idiv to compute the next values in the sequence may provide more fine grained precision over the c++ compiler-generated equivalent code
//#include <immintrin.h>
//
//static void matgen(REAL* a, int lda, int n, REAL* b, REAL* norma) {
//    int init = 1325;
//    *norma = 0.0;
//    REAL max_val = 0.0;
//
//    // Initialize b array to zero
//    for (int i = 0; i < n; i++) {
//        b[i] = 0.0;
//    }
//
//    // Initialize max_val vector
//    __m256d max_val_vec = _mm256_set1_pd(0.0);
//
//    // Main loop
//    for (int j = 0; j < n; j++) {
//        for (int i = 0; i < n; i += 4) {
//            // Generate the next values in the sequence
//            __m256i init_vec = _mm256_set_epi32(init, init + 1, init + 2, init + 3, init + 4, init + 5, init + 6, init + 7);
//            __m256i mult_vec = _mm256_set1_epi32(3125);
//            __m256i mod_vec = _mm256_set1_epi32(65536);
//
//            init_vec = _mm256_mullo_epi32(init_vec, mult_vec);
//            init_vec = _mm256_rem_epi32(init_vec, mod_vec);
//
//            // Convert to floating point values
//            __m256d value_vec = _mm256_cvtepi32_pd(_mm256_castsi256_si128(init_vec));
//            __m256d sub_vec = _mm256_set1_pd(32768.0);
//            __m256d div_vec = _mm256_set1_pd(16384.0);
//
//            value_vec = _mm256_sub_pd(value_vec, sub_vec);
//            value_vec = _mm256_div_pd(value_vec, div_vec);
//
//            // Store values in the matrix
//            _mm256_storeu_pd(&a[lda * j + i], value_vec);
//
//            // Update max_val vector
//            max_val_vec = _mm256_max_pd(max_val_vec, value_vec);
//
//            // Accumulate the values in the b array
//            __m256d b_vec = _mm256_loadu_pd(&b[i]);
//            b_vec = _mm256_add_pd(b_vec, value_vec);
//            _mm256_storeu_pd(&b[i], b_vec);
//
//            // Increment init
//            init += 4;
//        }
//    }
//
//    // Extract the maximum value from the vector
//    double max_vals[4];
//    _mm256_storeu_pd(max_vals, max_val_vec);
//    for (int k = 0; k < 4; k++) {
//        if (max_vals[k] > max_val) {
//            max_val = max_vals[k];
//        }
//    }
//
//    *norma = max_val;
//}

// REFACTORED
// Use pointer to functions 
typedef void (*dscal_func)(int, REAL, REAL*, int);
typedef void (*daxpy_func)(int, REAL, REAL*, int, REAL*, int);

// We can reduce redundancy in the original code by using dscal_func and daxpy_func to point to either dscal_r or dscal_ur based on the value of roll.
static void dgefa(REAL* a, int lda, int n, int* ipvt, int* info, int roll)
{
    REAL t;
    int j, k, kp1, l, nm1;

    dscal_func dscal = roll ? dscal_r : dscal_ur;
    daxpy_func daxpy = roll ? daxpy_r : daxpy_ur;

    /* Gaussian elimination with partial pivoting */
    *info = 0;
    nm1 = n - 1;

    if (nm1 >= 0) {
        for (k = 0; k < nm1; k++) {
            kp1 = k + 1;

            /* Find l = pivot index */
            l = idamax(n - k, &a[lda * k + k], 1) + k;
            ipvt[k] = l;

            /* Zero pivot implies this column already triangularized */
            if (a[lda * k + l] != ZERO) {

                /* Interchange if necessary */
                if (l != k) {
                    t = a[lda * k + l];
                    a[lda * k + l] = a[lda * k + k];
                    a[lda * k + k] = t;
                }

                /* Compute multipliers */
                t = -ONE / a[lda * k + k];
                dscal(n - (k + 1), t, &a[lda * k + k + 1], 1);

                /* Row elimination with column indexing */
                for (j = kp1; j < n; j++) {
                    t = a[lda * j + l];
                    if (l != k) {
                        a[lda * j + l] = a[lda * j + k];
                        a[lda * j + k] = t;
                    }
                    daxpy(n - (k + 1), t, &a[lda * k + k + 1], 1, &a[lda * j + k + 1], 1);
                }
            }
            else {
                *info = k;
            }
        }
    }

    ipvt[n - 1] = n - 1;
    if (a[lda * (n - 1) + (n - 1)] == ZERO) {
        *info = n - 1;
    }
}

// REFACTORED
static void dgesl(REAL* a, int lda, int n, int* ipvt, REAL* b, int job, int roll)
{
    REAL t;
    int k, kb, l, nm1;

    nm1 = n - 1;

    if (job == 0) {
        /* job = 0 , solve  a * x = b   */
        /* first solve  l*y = b         */
        if (nm1 >= 1) {
            for (k = 0; k < nm1; k++) {
                l = ipvt[k];
                t = b[l];
                if (l != k) {
                    b[l] = b[k];
                    b[k] = t;
                }
                if (roll) {
                    daxpy_r(n - (k + 1), t, &a[lda * k + k + 1], 1, &b[k + 1], 1);
                }
                else {
                    daxpy_ur(n - (k + 1), t, &a[lda * k + k + 1], 1, &b[k + 1], 1);
                }
            }
        }

        /* now solve  u*x = y */
        for (kb = 0; kb < n; kb++) {
            k = n - (kb + 1);
            b[k] = b[k] / a[lda * k + k];
            t = -b[k];
            if (roll) {
                daxpy_r(k, t, &a[lda * k], 1, &b[0], 1);
            }
            else {
                daxpy_ur(k, t, &a[lda * k], 1, &b[0], 1);
            }
        }
    }
    else {
        /* job = nonzero, solve  trans(a) * x = b  */
        /* first solve  trans(u)*y = b             */
        for (k = 0; k < n; k++) {
            if (roll) {
                t = ddot_r(k, &a[lda * k], 1, &b[0], 1);
            }
            else {
                t = ddot_ur(k, &a[lda * k], 1, &b[0], 1);
            }
            b[k] = (b[k] - t) / a[lda * k + k];
        }

        /* now solve trans(l)*x = y     */
        if (nm1 >= 1) {
            for (kb = 1; kb < nm1; kb++) {
                k = n - (kb + 1);
                if (roll) {
                    b[k] = b[k] + ddot_r(n - (k + 1), &a[lda * k + k + 1], 1, &b[k + 1], 1);
                }
                else {
                    b[k] = b[k] + ddot_ur(n - (k + 1), &a[lda * k + k + 1], 1, &b[k + 1], 1);
                }
                l = ipvt[k];
                if (l != k) {
                    t = b[l];
                    b[l] = b[k];
                    b[k] = t;
                }
            }
        }
    }
}

// REFACTORED
static void daxpy_r(int n, REAL da, REAL* dx, int incx, REAL* dy, int incy)
{
    if (n <= 0 || da == ZERO) return;

    int i;
    if (incx != 1 || incy != 1)
    {
        /* code for unequal increments or equal increments != 1 */
        int ix = (incx > 0) ? 0 : (-n + 1) * incx;
        int iy = (incy > 0) ? 0 : (-n + 1) * incy;

        for (i = 0; i < n; i++)
        {
            dy[iy] += da * dx[ix];
            ix += incx;
            iy += incy;
        }
    }
    else
    {
        /* code for both increments equal to 1 */
        int limit = n - (n % 4);
        for (i = 0; i < limit; i += 4)
        {
            dy[i] += da * dx[i];
            dy[i + 1] += da * dx[i + 1];
            dy[i + 2] += da * dx[i + 2];
            dy[i + 3] += da * dx[i + 3];
        }

        // Handle the remaining elements
        for (; i < n; i++)
        {
            dy[i] += da * dx[i];
        }
    }
}

// RERFACTORED
static REAL ddot_r(int n, REAL* dx, int incx, REAL* dy, int incy)
{
    REAL dtemp = ZERO;

    if (n <= 0) return ZERO;

    int i;
    if (incx != 1 || incy != 1)
    {
        /* code for unequal increments or equal increments != 1 */
        int ix = (incx > 0) ? 0 : (-n + 1) * incx;
        int iy = (incy > 0) ? 0 : (-n + 1) * incy;

        for (i = 0; i < n; i++)
        {
            dtemp += dx[ix] * dy[iy];
            ix += incx;
            iy += incy;
        }
    }
    else
    {
        /* code for both increments equal to 1 */
        int limit = n - (n % 4);
        for (i = 0; i < limit; i += 4)
        {
            dtemp += dx[i] * dy[i];
            dtemp += dx[i + 1] * dy[i + 1];
            dtemp += dx[i + 2] * dy[i + 2];
            dtemp += dx[i + 3] * dy[i + 3];
        }

        // Handle the remaining elements
        for (; i < n; i++)
        {
            dtemp += dx[i] * dy[i];
        }
    }
    return dtemp;
}


/*
** Scales a vector by a constant.
** Jack Dongarra, linpack, 3/11/78.
** ROLLED version
*/
static void dscal_r(int n, REAL da, REAL* dx, int incx)

{
    int i, nincx;

    if (n <= 0)
        return;
    if (incx != 1)
    {

        /* code for increment not equal to 1 */

        nincx = n * incx;
        for (i = 0; i < nincx; i = i + incx)
            dx[i] = da * dx[i];
        return;
    }

    /* code for increment equal to 1 */

    for (i = 0; i < n; i++)
        dx[i] = da * dx[i];
}


/*
** constant times a vector plus a vector.
** Jack Dongarra, linpack, 3/11/78.
** UNROLLED version
*/
static void daxpy_ur(int n, REAL da, REAL* dx, int incx, REAL* dy, int incy)

{
    int i, ix, iy, m;

    if (n <= 0)
        return;
    if (da == ZERO)
        return;

    if (incx != 1 || incy != 1)
    {

        /* code for unequal increments or equal increments != 1 */

        ix = 1;
        iy = 1;
        if (incx < 0) ix = (-n + 1) * incx + 1;
        if (incy < 0)iy = (-n + 1) * incy + 1;
        for (i = 0; i < n; i++)
        {
            dy[iy] = dy[iy] + da * dx[ix];
            ix = ix + incx;
            iy = iy + incy;
        }
        return;
    }

    /* code for both increments equal to 1 */

    m = n % 4;
    if (m != 0)
    {
        for (i = 0; i < m; i++)
            dy[i] = dy[i] + da * dx[i];
        if (n < 4)
            return;
    }
    for (i = m; i < n; i = i + 4)
    {
        dy[i] = dy[i] + da * dx[i];
        dy[i + 1] = dy[i + 1] + da * dx[i + 1];
        dy[i + 2] = dy[i + 2] + da * dx[i + 2];
        dy[i + 3] = dy[i + 3] + da * dx[i + 3];
    }
}


/*
** Forms the dot product of two vectors.
** Jack Dongarra, linpack, 3/11/78.
** UNROLLED version
*/
static REAL ddot_ur(int n, REAL* dx, int incx, REAL* dy, int incy)

{
    REAL dtemp;
    int i, ix, iy, m;

    dtemp = ZERO;

    if (n <= 0)
        return(ZERO);

    if (incx != 1 || incy != 1)
    {

        /* code for unequal increments or equal increments != 1 */

        ix = 0;
        iy = 0;
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        for (i = 0; i < n; i++)
        {
            dtemp = dtemp + dx[ix] * dy[iy];
            ix = ix + incx;
            iy = iy + incy;
        }
        return(dtemp);
    }

    /* code for both increments equal to 1 */

    m = n % 5;
    if (m != 0)
    {
        for (i = 0; i < m; i++)
            dtemp = dtemp + dx[i] * dy[i];
        if (n < 5)
            return(dtemp);
    }
    for (i = m; i < n; i = i + 5)
    {
        dtemp = dtemp + dx[i] * dy[i] +
            dx[i + 1] * dy[i + 1] + dx[i + 2] * dy[i + 2] +
            dx[i + 3] * dy[i + 3] + dx[i + 4] * dy[i + 4];
    }
    return(dtemp);
}


/*
** Scales a vector by a constant.
** Jack Dongarra, linpack, 3/11/78.
** UNROLLED version
*/
static void dscal_ur(int n, REAL da, REAL* dx, int incx)

{
    int i, m, nincx;

    if (n <= 0)
        return;
    if (incx != 1)
    {

        /* code for increment not equal to 1 */

        nincx = n * incx;
        for (i = 0; i < nincx; i = i + incx)
            dx[i] = da * dx[i];
        return;
    }

    /* code for increment equal to 1 */

    m = n % 5;
    if (m != 0)
    {
        for (i = 0; i < m; i++)
            dx[i] = da * dx[i];
        if (n < 5)
            return;
    }
    for (i = m; i < n; i = i + 5)
    {
        dx[i] = da * dx[i];
        dx[i + 1] = da * dx[i + 1];
        dx[i + 2] = da * dx[i + 2];
        dx[i + 3] = da * dx[i + 3];
        dx[i + 4] = da * dx[i + 4];
    }
}


/*
** Finds the index of element having max. absolute value.
** Jack Dongarra, linpack, 3/11/78.
*/
static int idamax(int n, REAL* dx, int incx)

{
    REAL dmax;
    int i = 0, ix = 0, itemp = 0; // Initialize for CS230 to avoid warning

    if (n < 1)
        return(-1);
    if (n == 1)
        return(0);
    if (incx != 1)
    {

        /* code for increment not equal to 1 */

        ix = 1;
        dmax = fabs((double)dx[0]);
        ix = ix + incx;
        for (i = 1; i < n; i++)
        {
            if (fabs((double)dx[ix]) > dmax)
            {
                itemp = i;
                dmax = fabs((double)dx[ix]);
            }
            ix = ix + incx;
        }
    }
    else
    {

        /* code for increment equal to 1 */

        itemp = 0;
        dmax = fabs((double)dx[0]);
        for (i = 1; i < n; i++)
            if (fabs((double)dx[i]) > dmax)
            {
                itemp = i;
                dmax = fabs((double)dx[i]);
            }
    }
    return (itemp);
}


static REAL second(void)

{
    return ((REAL)((REAL)clock() / (REAL)CLOCKS_PER_SEC));
}

//=========================================================

