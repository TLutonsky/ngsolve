/*
  The singular value decomposition: Anatomy of optimizing an algorithm for extreme scale
  J Dongarra, M Gates, A Haidar, J Kurzak, P Luszczek, S Tomov, ...
  SIAM review 60 (4), 808-865 (2017)

  L. Hogben (ed): Handbook of Linear Algebra, Chap 45


  * use the two-step bidiagonalization using Householder reflections from both sides
  * bulge chasing (by B. Lang), an  O(N*N*bs) algorithm for diagonalization
  * bisection for singular values
  * twisted factorization (RRR relatively robust representation)

  TODO:
  MRRR ... multiple RRR
  degegenerate cases (with intermediate zeros)
*/


#include <bla.hpp>

namespace ngbla
{

  /*
    int dgesvd_(char *jobu, char *jobvt, integer *m, integer *n,
    doublereal *a, integer *lda, doublereal *s, doublereal *u, integer *
    ldu, doublereal *vt, integer *ldvt, doublereal *work, integer *lwork,
    integer *info);
  */

  void LapackSVD (SliceMatrix<> A,
                  SliceMatrix<double, ColMajor> U,
                  SliceMatrix<double, ColMajor> V)
  {
    static Timer t("LapackSVD"); RegionTimer reg(t);
    ngbla::integer m = A.Width(), n = A.Height();
    // Matrix<> U(m), V(n);
    Vector<> S(min(n,m));
    Array<double> work(n*m+100);
    ngbla::integer info;
    char jobu = 'A', jobv = 'A';
    ngbla::integer lda = A.Dist(), ldu = U.Dist(), ldv = V.Dist();
    ngbla::integer lwork = work.Size();

    dgesvd_ ( &jobu, &jobv, &m, &n, A.Data(), &lda,
              S.Data(),
              U.Data(), &ldu, V.Data(), &ldv,
              work.Data(), &lwork,
              &info);
    // cout << "info = " << info << endl;
    // if (n <= 100)
    // cout << "S = " << S << endl;
    A.Diag(0) = S;
  }

  void LapackSVD (SliceMatrix<double,ColMajor> A,
                  SliceMatrix<double, ColMajor> U,
                  SliceMatrix<double, ColMajor> V)
  {
    return LapackSVD(Trans(A), V, U);
  }


  /******************** BIDIAGONALIZATION ********************/
  template <ORDERING OH, ORDERING OM>
  void ApplyBandHouseholderReflections (size_t bs, SliceMatrix<double,OH> H, SliceMatrix<double,OM> M)
  {
    /*
    size_t n = H.Width();
    for (size_t i = 0; i+bs < n; i++)
      H.Col(i).Range(i+bs, min(n, i+2*bs)) = 0;
    for (size_t i = 0; i < H.Width(); i += bs)
      {
        size_t next = min(i+bs, n);
        MultiHouseholderReflection Hv(Trans(H.Cols(i,next).Rows(i,min(i+2*bs,n))));
        Hv.Mult(M.Rows(i,min(i+2*bs,n)));
      }
    */
    // if (bs > 96) throw Exception("BandMatrix: band to big");
    // double mem[96*192];
    ArrayMem<double,48*96> mem(2*bs*bs);
    ArrayMem<double,48*48> memh(bs*bs);
    size_t n = H.Width();

    for (size_t i = 0; i < H.Width(); i += bs)
      {
        IntRange cols(i, min(i+bs,n));
        IntRange rows(i, min(i+2*bs,n));
        FlatMatrix<double,OH> tmp (rows.Size(), cols.Size(),mem.Data());
        tmp = H.Cols(cols).Rows(rows);
        for (size_t i = 0; i+bs < tmp.Height(); i++)
          tmp.Col(i).Range(i+bs, tmp.Height()) = 0;
        MultiHouseholderReflectionMem Hv(SliceMatrix(Trans(tmp)), memh.Data());
        Hv.Mult(M.Rows(rows));
      }
  }

  template <ORDERING ORD>
  void ReduceBiDiagonal (SliceMatrix<double,ORD> A,
                         SliceMatrix<double, ColMajor> U1, SliceMatrix<double, ColMajor> V1)
  {
    static Timer ttrig("householder-triangular"); RegionTimer reg(ttrig);
    // static Timer tuv("householder-triangular, transform uv");

    size_t m = A.Height();
    size_t n = A.Width();
    size_t minnm = min(n,m);


    /*
    // no blocking at all
    Vector<> v2(max(n,m));
    for (int i = 0; i < n; i++)
    {
    auto Arest = A.Rows(i,m).Cols(i,n);
    auto vrest = v2.Range(i,m);
    CalcHouseholderVector(Arest.Col(0), vrest);
    HouseholderReflection (vrest).Mult(Arest);
    HouseholderReflection (vrest).Mult(Trans(U1).Rows(i,m));

    if (i+1 < n)
    {
    auto Arest = A.Rows(i,m).Cols(i+1,n);
    auto vrest = v2.Range(i+1,n);
    CalcHouseholderVector(Arest.Row(0), vrest);
    HouseholderReflection (vrest).Mult(Trans(Arest));
    HouseholderReflection (vrest).Mult(Trans(V1).Rows(i+1,n));
    }
    }
    */

    /*
    // no blocking of A, but blocking of U and V
    constexpr size_t bs = 16;
    Matrix<> mvc(bs, m);
    Matrix<> mvr(bs, n);
    for (int i1 = 0; i1 < min(n,m); i1 += bs)
    {
    int bs1 = min( min(n-i1, m-i1), bs);
    int bs2 = 0;
    mvc = 0.0;
    mvr = 0.0;

    for (int i2 = 0; i2 < bs1; i2++)
    {
    int i = i1+i2;
    auto Arest = A.Rows(i,m).Cols(i,n);
    CalcHouseholderVector(Arest.Col(0), mvc.Row(i2).Range(i,m));
    HouseholderReflection (mvc.Row(i2).Range(i,m)).Mult(Arest);

    if (i+1 < n)
    {
    bs2 = i2+1;
    auto Arest = A.Rows(i,m).Cols(i+1,n);
    CalcHouseholderVector (Arest.Row(0), mvr.Row(i2).Range(i+1,n));
    HouseholderReflection (mvr.Row(i2).Range(i+1,n)).Mult(Trans(Arest));
    }
    }

    MultiHouseholderReflection (mvc.Rows(0,bs1).Cols(i1,m)).Mult (Trans(U1).Rows(i1,m));
    if (bs2 > 0)
    MultiHouseholderReflection (mvr.Rows(0,bs2).Cols(i1+1,n)).Mult (Trans(V1).Rows(i1+1,n));
    }
    // cout << "A bidiag = " << endl << Truncate (A) << endl;
    */

    VectorMem<100> hv(max(n,m));
    for (size_t i = 0; i < minnm; i++)
      {
        auto Acol = A.Cols(i,i+1).Rows(i,m);  // a matrix
        double signed_norm = CalcHouseholderVectorInPlace (Acol.Col(0));
        hv.Range(i,m) = Acol.Col(0);
        Acol(0,0) = signed_norm;

        // MultiHouseholderReflection H(Trans(Acol));
        HouseholderReflection H(hv.Range(i,m));     // more efficient
        H.Mult (A.Rows(i,m).Cols(i+1,n));

        if (i+1 < n)
          {
            auto Arow = A.Rows(i,i+1).Cols(i+1,n);  // a matrix
            double signed_norm = CalcHouseholderVectorInPlace (Arow.Row(0));
            hv.Range(i+1,n) = Arow.Row(0);
            Arow(0,0) = signed_norm;

            // MultiHouseholderReflection H(Arow);
            HouseholderReflection H(hv.Range(i+1,n));   // more efficient
            H.Mult (Trans (A.Rows(i+1,m).Cols(i+1,n)));
          }
      }

    // RegionTimer ruv(tuv);
    static Timer tid("setid");
    U1 = Identity(m);
    ApplyHouseholderReflections (A, Trans(U1));

    tid.Start();
    V1 = Identity(n);
    tid.Stop();
    if (n > 1)
      ApplyHouseholderReflections (Trans(A.Cols(1,n).Rows(min(n-1,m))), Trans(V1).Rows(1,n));

    for (size_t i = 0; i < minnm; i++)
      A.Col(i).Range(i+1, m) = 0.0;
    for (size_t i = 1; i < minnm; i++)
      A.Col(i).Range(0, i-1) = 0.0;
  }

  /*
    n = 1000, bs 36:
    band:   114
    bulge:   39
    mat U/V  171/163

    n = 1000, bs 12:
    band: 226
    bulge: 22
    mat U/V: 241/235
   */


  template <ORDERING OA, ORDERING ORD>
  void ReduceBiDiagonalBlocked (SliceMatrix<double, OA> A,
                                SliceMatrix<double, ORD> U1, SliceMatrix<double, ORD> V1)
  {
    /*
      Block-factorization.

      A parallel algorithm for reducing symmetric banded matrices to tridiagonal form
      Bruno Lang - SIAM Journal on Scientific Computing, 1993 - SIAM
    */

    // cout << "ReducedBiDiagnoalBlocked" << endl;
    static Timer ttrig("householder-block-triangular");
    static Timer tUV("householder-block-triangular, transform U1,V1");
    static Timer tU("transform U1");
    static Timer tV("transform V1");
    static Timer tbulgechasing("bulge chasing");
    ttrig.Start();

    size_t m = A.Height();
    size_t n = A.Width();
    Matrix A_orig = A; // for testint only
    constexpr size_t bs = 36;  // orig: 32


    for (size_t i1 = 0; i1 < min(n,m); i1 += bs)
      {
        size_t bs1 = min( min(n-i1, m-i1), bs);
        for (size_t i2 = 0; i2 < bs1; i2++)
          {
            size_t i = i1+i2;

            auto Apanel = A.Rows(i,m).Cols(i,i1+bs1);
            double signed_norm = CalcHouseholderVectorInPlace (Apanel.Col(0));
            Apanel(0,0) = signed_norm;

            MultiHouseholderReflection H(Trans(Apanel.Cols(0,1)));
            // HouseholderReflection1 H(Apanel.Col(0));

            H.Mult (Apanel.Cols(1, Apanel.Width()));
          }

        if (i1+bs1 < n)
          {
            MultiHouseholderReflection H(Trans(A.Cols(i1, i1+bs1).Rows(i1, m)));
            H.Mult (A.Rows(i1,m).Cols(i1+bs1,n));
          }

        for (int i2 = 0; i2 < bs1; i2++)
          {
            size_t i = i1 + i2;
            if (i+bs1 < n)
              {
                auto Apanel = A.Rows(i,i1+bs1).Cols(i+bs1, n);
                double signed_norm = CalcHouseholderVectorInPlace (Apanel.Row(0));
                Apanel(0,0) = signed_norm;

                MultiHouseholderReflection H(Apanel.Rows(0,1));
                // HouseholderReflection1 H(Apanel.Row(0));
                H.Mult (Trans(Apanel.Rows(1, Apanel.Height())));
              }
          }

        size_t bs2 = min(bs1, n-i1-bs1);
        if (bs2 > 0)
          if (i1+bs1 < m)
            {
              MultiHouseholderReflection H(A.Rows(i1, i1+bs2).Cols(i1+bs1, n));
              H.Mult (Trans(A.Rows(i1+bs1,m).Cols(i1+bs1,n)));
            }
      }

    ttrig.Stop();


    /*
      Matrix<> Abandmem(minnm, minnm);
      Abandmem = 0.0;
      // simulate a band-matrix by reducing dist:
      SliceMatrix<> Aband(minnm, minnm, min(3*bs, minnm), Abandmem.Data());
      for (int i = 0; i < min(n, bs+1); i++)
      Aband.Diag(i) = A.Rows(minnm).Cols(minnm).Diag(i);
    */
    // auto Aband = A.Rows(minnm).Cols(minnm);

    size_t minnm = min(n,m);

    Matrix Aband = A.Rows(minnm).Cols(minnm);

    for (size_t i = 0; i < minnm; i++)
      Aband.Col(i).Range(i+1, minnm) = 0.0;
    for (size_t i = bs; i < minnm; i++)
      Aband.Col(i).Range(0, i-bs) = 0.0;


    /*
      if (n <= 20)
      {
      cout << "householder, blocked = " << endl << Truncate(A) << endl;
      cout << "U^T A V = " << endl << Truncate( Matrix (Matrix(Trans(U1)*A_orig) * V1)) << endl;
      }
    */


    // now do the bulge chasing
    // tbulgechasing.Start();
    /*
      Vector<> tmp(bs);
      for (int i = 0; i < n-1; i++)
      {
      // first vector
      IntRange cols(i+1, min(n, i+bs+1));
      auto x = A.Row(i).Range(cols);
      CalcHouseholderVector (x, tmp.Range(x.Size()));

      IntRange rows(i, min(m,i+bs+1));
      HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(A.Rows(rows).Cols(cols)));
      HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(V1.Cols(cols)));

      for (int i1 = i+1; i1 < n; i1 += bs)
      {
      {
      IntRange rows(i1, min(n, i1+bs));
      auto x = A.Col(i1).Range(rows);
      CalcHouseholderVector (x, tmp.Range(x.Size()));

      IntRange cols(i1, min(n, i1+2*bs));
      HouseholderReflection (tmp.Range(x.Size())).Mult (A.Rows(rows).Cols(cols));
      HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(U1.Cols(rows)));
      }

      if (i1+bs < n)
      {
      IntRange cols(i1+bs, min(n, i1+2*bs));
      auto x = A.Row(i1).Range(cols);
      CalcHouseholderVector (x, tmp.Range(x.Size()));

      IntRange rows(i1, min(n, i1+2*bs));
      HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(A.Rows(rows).Cols(cols)));
      HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(V1.Cols(cols)));
      }
      }
      }
    */



    /*
      TODO: store the bulge-chasing reflections in triangular matrix
      write banded - Multihouseholder
    */


    /*
      constexpr size_t bcbs = 16;   // bulge chasing block size
      Vector<> tmp(bs);
      int maxblocks = n/bs+1;
      Matrix<> reflect_bcU(bcbs * maxblocks, bs+bcbs-1);
      Matrix<> reflect_bcV(bcbs * maxblocks, bs+bcbs-1);
      Array<IntRange> ranges;

      for (int bi = 0; bi < n-1; bi += bcbs)
      {
      int bsi = min(bcbs, n-1-bi);
      reflect_bcU = 0.0;
      reflect_bcV = 0.0;

      for (int ii = 0; ii < bsi; ii++)
      {
      int i = bi + ii;
      // first vector
      IntRange cols(i+1, min(n, i+bs+1));
      auto x = Aband.Row(i).Range(cols);
      CalcHouseholderVector (x, tmp.Range(x.Size()));
      reflect_bcV.Row(0*bcbs+ii).Range(ii, ii+cols.Size()) = tmp.Range(x.Size());

      IntRange rows(i, min(m,i+bs+1));
      HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(Aband.Rows(rows).Cols(cols)));
      // HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(V1.Cols(cols)));
      int nr = 0;
      for (int i1 = i+1; i1 < n; i1 += bs, nr++)
      {
      {
      IntRange rows(i1, min(n, i1+bs));
      auto x = Aband.Col(i1).Range(rows);
      CalcHouseholderVector (x, tmp.Range(x.Size()));
      reflect_bcU.Row(nr*bcbs+ii).Range(ii, ii+rows.Size()) = tmp.Range(x.Size());

      IntRange cols(i1, min(n, i1+2*bs));
      HouseholderReflection (tmp.Range(x.Size())).Mult (Aband.Rows(rows).Cols(cols));
      // HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(U1.Cols(rows)));
      }

      if (i1+bs < n)
      {
      IntRange cols(i1+bs, min(n, i1+2*bs));
      auto x = Aband.Row(i1).Range(cols);
      CalcHouseholderVector (x, tmp.Range(x.Size()));
      reflect_bcV.Row((nr+1)*bcbs+ii).Range(ii, ii+cols.Size()) = tmp.Range(x.Size());

      IntRange rows(i1, min(n, i1+2*bs));
      HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(Aband.Rows(rows).Cols(cols)));
      // HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(V1.Cols(cols)));
      }
      }
      }

      ranges.SetSize(0);
      for (int i1 = bi+1; i1 < n; i1 += bs)
      ranges.Append (IntRange(i1, min(n,i1+bs+bcbs-1)));
      for (int i = ranges.Size()-1; i >= 0; i--)
      {
      int bcbsi = min(bcbs, ranges[i].Size());
      MultiHouseholderReflection H(reflect_bcU.Rows(i*bcbs, i*bcbs+bcbsi).Cols(ranges[i].Size()));
      H.Mult (Trans(U1.Cols(ranges[i])));
      }
      for (int i = ranges.Size()-1; i >= 0; i--)
      {
      int bcbsi = min(bcbs, ranges[i].Size());
      MultiHouseholderReflection (reflect_bcV.Rows(i*bcbs, i*bcbs+bcbsi).Cols(ranges[i].Size())).Mult (Trans(V1.Cols(ranges[i])));
      }
      }
    */



    // constexpr size_t bcbs = 16;   // bulge chasing block size
    Vector<> tmp(bs);
    // int maxblocks = n/bs+1;
    // Matrix<> reflect_bcU(bcbs * maxblocks, bs+bcbs-1);
    // Matrix<> reflect_bcV(bcbs * maxblocks, bs+bcbs-1);
    // Array<IntRange> ranges;

    // Array<tuple<int,int,int>> refu;
    // Array<tuple<int,int,int>> refv;

    static Timer tb1 ("bulge chasing - transform band");
    tb1.Start();

    for (size_t i = 0; i+1 < n; i++)
      {
        // first vector
        IntRange cols(i+1, min(n, i+bs+1));
        auto x = Aband.Row(i).Range(cols);

        CalcHouseholderVector (x, tmp.Range(x.Size()));
        // refv.Append ( tuple(i,cols.First(),cols.Size() ));

        IntRange rows(i, min(m,i+bs+1));
        HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(Aband.Rows(rows).Cols(cols)));
        Aband.Row(i).Range(cols.First()+1, cols.Next()) = tmp.Range(1,x.Size());

        int nr = 0;
        for (int i1 = i+1; i1 < n; i1 += bs, nr++)
          {
            {
              IntRange rows(i1, min(n, i1+bs));
              auto x = Aband.Col(i1).Range(rows);
              CalcHouseholderVector (x, tmp.Range(x.Size()));

              Aband.Col(i).Range(rows) = tmp.Range(x.Size());
              // refu.Append ( tuple(i1,i,rows.Size() ));
              IntRange cols(i1, min(n, i1+2*bs));
              HouseholderReflection (tmp.Range(x.Size())).Mult (Aband.Rows(rows).Cols(cols));
            }

            if (i1+bs < n)
              {
                IntRange cols(i1+bs, min(n, i1+2*bs));
                auto x = Aband.Row(i1).Range(cols);
                CalcHouseholderVector (x, tmp.Range(x.Size()));

                Aband.Row(i).Range(cols) = tmp.Range(x.Size());
                // refv.Append ( tuple(i,cols.First(),cols.Size() ));

                IntRange rows(i1, min(n, i1+2*bs));
                HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(Aband.Rows(rows).Cols(cols)));
              }
          }
      }

    tb1.Stop();



    tUV.Start();

    // U1 = Identity(m);
    // V1 = Identity(n);
    U1 = 0.0; U1.Diag() = 1;
    V1 = 0.0; V1.Diag() = 1;

    tU.Start();
    // ApplyHouseholderReflections (A, Trans(U1));
    // ApplyHouseholderReflectionsTrans (A, U1);
    tU.Stop();
    // sum i=1..n : 2(m-i)*m  = 2 (n*m*m-n*n/2*m) = nm(2m-n)
    tU.AddFlops(n*m*(2*m-n));

    tUV.Stop();


    static Timer tb2 ("bulge chasing - mult U");
    static Timer tb3 ("bulge chasing - mult V");
    tb2.AddFlops (n*n*n);
    tb3.AddFlops (n*n*n);
    tb2.Start();

    // startrow is   1 + k bs,  maximal n-1
    // int startrow = 1;
    // while (startrow+bs < n) startrow += bs;
    int startrow = n-2;
    startrow -= startrow % bs;
    startrow++;
    while (startrow > 0)
      {
        IntRange rest(startrow, n);
        auto sub = Aband.Rows(rest).Cols(0, n-startrow);
        ApplyBandHouseholderReflections (bs, sub, Trans(U1.Cols(rest)).Cols(startrow,m));

        ApplyHouseholderReflectionsTrans (A.Cols(startrow, min(startrow+bs,n)).Rows(startrow,m), U1.Rows(startrow,m).Cols(startrow,m));
        startrow -= bs;
      }
    // ApplyHouseholderReflectionsTrans (A.Cols(0, min(startrow+bs,n)), U1);
    ApplyHouseholderReflectionsTrans (A.Cols(0, 1), U1);

    tb2.Stop();

    tb3.Start();

    startrow = 1;
    while (startrow+bs < n) startrow += bs;
    while (startrow > 0)
      {
        IntRange rest(startrow, n);
        auto sub = Aband.Cols(rest).Rows(0, n-startrow);
        ApplyBandHouseholderReflections (bs, Trans(sub), Trans(V1.Cols(rest)).Cols(rest) );

        if (startrow+bs<n)
          ApplyHouseholderReflectionsTrans (Trans(A.Rows(startrow, min(startrow+bs, n-bs)).Cols(startrow+bs, n)), V1.Rows(startrow+bs,n).Cols(rest));

        startrow -= bs;
      }
    // ApplyHouseholderReflectionsTrans (A.Cols(0, min(startrow+bs,n)), U1);
    ApplyHouseholderReflectionsTrans (Trans(A.Rows(0, 1).Cols(bs, n)), V1.Rows(bs,n));
    tb3.Stop();

    tV.Start();
    // if (n > bs)
    // ApplyHouseholderReflectionsTrans (Trans(A.Cols(bs,n).Rows(min(n-bs,m))), V1.Rows(bs,n));
    tV.Stop();
    tV.AddFlops (sqr(n-bs)*n);

    // tbulgechasing.Stop();

    // A.Rows(minnm).Cols(minnm) = Aband;
    A.Diag(0) = Aband.Diag(0);
    A.Diag(1) = Aband.Diag(1);

    // if (n < 20)
    // cout << "after bulge-chasing: " << endl << Truncate(A) << endl;
  }


  // same as void ReduceBiDiagonal (SliceMatrix<double,ORD>, SliceMatrix<double, ColMajor>, SliceMatrix<double, ColMajor>)
  // without applying householders to U and V
  template <ORDERING ORD>
  void ReduceBiDiagonal (SliceMatrix<double,ORD> A)
  {
    // static Timer ttrig("householder-triangular"); RegionTimer reg(ttrig);
    // static Timer tuv("householder-triangular, transform uv");

    size_t m = A.Height();
    size_t n = A.Width();
    size_t minnm = min(n,m);

    VectorMem<100> hv(max(n,m));
    for (size_t i = 0; i < minnm; i++)
      {
        auto Acol = A.Cols(i,i+1).Rows(i,m);  // a matrix
        double signed_norm = CalcHouseholderVectorInPlace (Acol.Col(0));
        hv.Range(i,m) = Acol.Col(0);
        Acol(0,0) = signed_norm;

        // MultiHouseholderReflection H(Trans(Acol));
        HouseholderReflection H(hv.Range(i,m));     // more efficient
        H.Mult (A.Rows(i,m).Cols(i+1,n));

        if (i+1 < n)
          {
            auto Arow = A.Rows(i,i+1).Cols(i+1,n);  // a matrix
            double signed_norm = CalcHouseholderVectorInPlace (Arow.Row(0));
            hv.Range(i+1,n) = Arow.Row(0);
            Arow(0,0) = signed_norm;

            // MultiHouseholderReflection H(Arow);
            HouseholderReflection H(hv.Range(i+1,n));   // more efficient
            H.Mult (Trans (A.Rows(i+1,m).Cols(i+1,n)));
          }
      }
  }

  // Version for LowRankSVD
  template <ORDERING OA, ORDERING ORD>
  void ReduceBiDiagonalBlocked (SliceMatrix<double, OA> A, SliceMatrix<double, ColMajor> Aband)
  {
    /*
      Block-factorization.

      A parallel algorithm for reducing symmetric banded matrices to tridiagonal form
      Bruno Lang - SIAM Journal on Scientific Computing, 1993 - SIAM
    */

    // cout << "ReducedBiDiagnoalBlocked" << endl;
    static Timer ttrig("householder-block-triangular");
    // ttrig.Start();

    size_t m = A.Height();
    size_t n = A.Width();
    //Matrix A_orig = A; // for testing only
    constexpr size_t bs = 36;  // orig: 32


    for (size_t i1 = 0; i1 < min(n,m); i1 += bs)
      {
        size_t bs1 = min( min(n-i1, m-i1), bs);
        for (size_t i2 = 0; i2 < bs1; i2++)
          {
            size_t i = i1+i2;

            auto Apanel = A.Rows(i,m).Cols(i,i1+bs1);
            double signed_norm = CalcHouseholderVectorInPlace (Apanel.Col(0));
            Apanel(0,0) = signed_norm;

            MultiHouseholderReflection H(Trans(Apanel.Cols(0,1)));
            //HouseholderReflection1   H(Apanel.Col(0));

            H.Mult (Apanel.Cols(1, Apanel.Width()));
          }

        if (i1+bs1 < n)
          {
            MultiHouseholderReflection H(Trans(A.Cols(i1, i1+bs1).Rows(i1, m)));
            H.Mult (A.Rows(i1,m).Cols(i1+bs1,n));
          }

        for (int i2 = 0; i2 < bs1; i2++)
          {
            size_t i = i1 + i2;
            if (i+bs1 < n)
              {
                auto Apanel = A.Rows(i,i1+bs1).Cols(i+bs1, n);
                double signed_norm = CalcHouseholderVectorInPlace (Apanel.Row(0));
                Apanel(0,0) = signed_norm;

                MultiHouseholderReflection H(Apanel.Rows(0,1));
                // HouseholderReflection1 H(Apanel.Row(0));
                H.Mult (Trans(Apanel.Rows(1, Apanel.Height())));
              }
          }

        size_t bs2 = min(bs1, n-i1-bs1);
        if (bs2 > 0)
          if (i1+bs1 < m)
            {
              MultiHouseholderReflection H(A.Rows(i1, i1+bs2).Cols(i1+bs1, n));
              H.Mult (Trans(A.Rows(i1+bs1,m).Cols(i1+bs1,n)));
            }
      }

    // ttrig.Stop();


    size_t minnm = min(n,m);

    Aband = A.Rows(minnm).Cols(minnm);

    for (size_t i = 0; i < minnm; i++)
      Aband.Col(i).Range(i+1, minnm) = 0.0;
    for (size_t i = bs; i < minnm; i++)
      Aband.Col(i).Range(0, i-bs) = 0.0;


    // constexpr size_t bcbs = 16;   // bulge chasing block size
    Vector<> tmp(bs);
    // int maxblocks = n/bs+1;
    // Matrix<> reflect_bcU(bcbs * maxblocks, bs+bcbs-1);
    // Matrix<> reflect_bcV(bcbs * maxblocks, bs+bcbs-1);
    // Array<IntRange> ranges;

    // Array<tuple<int,int,int>> refu;
    // Array<tuple<int,int,int>> refv;

    static Timer tb1 ("bulge chasing - transform band");
    // tb1.Start();

    for (size_t i = 0; i+1 < min(m,n); i++)
      {

        // first vector
        IntRange cols(i+1, min(min(m,n), i+bs+1));
        auto x = Aband.Row(i).Range(cols);

        CalcHouseholderVector (x, tmp.Range(x.Size()));

        // refv.Append ( tuple(i,cols.First(),cols.Size() ));

        IntRange rows(i, min(min(m,n),i+bs+1));
        HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(Aband.Rows(rows).Cols(cols)));

        Aband.Row(i).Range(cols.First()+1, cols.Next()) = tmp.Range(1,x.Size());

        int nr = 0;
        for (int i1 = i+1; i1 < n; i1 += bs, nr++)
          {

            {
              IntRange rows(i1, min(min(m,n), i1+bs));
              auto x = Aband.Col(i1).Range(rows);
              CalcHouseholderVector (x, tmp.Range(x.Size()));

              Aband.Col(i).Range(rows) = tmp.Range(x.Size());
              // refu.Append ( tuple(i1,i,rows.Size() ));
              IntRange cols(i1, min(min(m,n), i1+2*bs));
              HouseholderReflection (tmp.Range(x.Size())).Mult (Aband.Rows(rows).Cols(cols));

            }

            if (i1+bs < n)
              {
                IntRange cols(i1+bs, min(min(m,n), i1+2*bs));
                auto x = Aband.Row(i1).Range(cols);
                CalcHouseholderVector (x, tmp.Range(x.Size()));

                Aband.Row(i).Range(cols) = tmp.Range(x.Size());
                // refv.Append ( tuple(i,cols.First(),cols.Size() ));

                IntRange rows(i1, min(min(m,n), i1+2*bs));
                HouseholderReflection (tmp.Range(x.Size())).Mult (Trans(Aband.Rows(rows).Cols(cols)));
              }
          }
      }

    // tb1.Stop();

  }

  // apply householder transformations (stored in A) to U and V
  // used in LowRankSVD
  template <ORDERING ORD>
  void BidiagTransformationsSmall (SliceMatrix<double,ORD> A,
                 SliceMatrix<double, ColMajor> U1,
                 SliceMatrix<double, ColMajor> V1)
  {
    int m = A.Height();
    int n = A.Width();

    // RegionTimer ruv(tuv);
    // static Timer tUVHouse("CalcSVD apply householder to U, V"); // THOMPSN
    // tUVHouse.Start(); // THOMPSN
    U1 = Identity(m);
    ApplyHouseholderReflections (A, SliceMatrix(Trans(U1)));

    V1 = Identity(n);
    if (n > 1)
      ApplyHouseholderReflections (Trans(A.Cols(1,n).Rows(min(n-1,m))), SliceMatrix(Trans(V1).Rows(1,n)));
    // tUVHouse.Stop();

  }

  // apply householders (stored in A and Aband) to U1 and U2
  template <ORDERING ORD>
  void BidiagTransformationsBlocked (SliceMatrix<double,ORD> A,
                 SliceMatrix<double, ColMajor> Aband,
                 SliceMatrix<double, ColMajor> U1,
                 SliceMatrix<double, ColMajor> V1)
  {
    int m = A.Height();
    int n = A.Width();

    // U1 = Identity(m);
    // V1 = Identity(n);
    U1 = 0.0; U1.Diag() = 1;
    V1 = 0.0; V1.Diag() = 1;

    // static Timer tb2 ("bulge chasing - mult U");
    // static Timer tb3 ("bulge chasing - mult V");
    // tb2.AddFlops (n*n*n);
    // tb3.AddFlops (n*n*n);
    // tb2.Start();

    // startrow is   1 + k bs,  maximal n-1
    // int startrow = 1;
    // while (startrow+bs < n) startrow += bs;
    int bs = 36;
    int startrow = n-2;
    startrow -= startrow % bs;
    startrow++;
    while (startrow > 0)
      {
        IntRange rest(startrow, n);
        auto sub = Aband.Rows(rest).Cols(0, n-startrow);
        ApplyBandHouseholderReflections (bs, sub, Trans(U1.Cols(rest)).Cols(startrow,m));

        ApplyHouseholderReflectionsTrans (A.Cols(startrow, min(startrow+bs,n)).Rows(startrow,m), U1.Rows(startrow,m).Cols(startrow,m));
        startrow -= bs;
      }
    // ApplyHouseholderReflectionsTrans (A.Cols(0, min(startrow+bs,n)), U1);
    ApplyHouseholderReflectionsTrans (A.Cols(0, 1), U1);

    // tb2.Stop();

    // tb3.Start();

    startrow = 1;
    /* while (startrow+bs < n) startrow += bs; */
		startrow = 1 + n - n % bs;
    while (startrow > 0)
      {
        IntRange rest(startrow, n);
        auto sub = Aband.Cols(rest).Rows(0, n-startrow);
        ApplyBandHouseholderReflections (bs, Trans(sub), Trans(V1.Cols(rest)).Cols(rest) );

        if (startrow+bs<n)
          ApplyHouseholderReflectionsTrans (Trans(A.Rows(startrow, min(startrow+bs, n-bs)).Cols(startrow+bs, n)), V1.Rows(startrow+bs,n).Cols(rest));

        startrow -= bs;
      }
    // ApplyHouseholderReflectionsTrans (A.Cols(0, min(startrow+bs,n)), U1);
    ApplyHouseholderReflectionsTrans (Trans(A.Rows(0, 1).Cols(bs, n)), V1.Rows(bs,n));
    // tb3.Stop();

    A.Diag(0) = Aband.Diag(0);
    A.Diag(1) = Aband.Diag(1);
  }



  /* ***************** Bi-diagonal SVD by bisection ************ */

  /*
    Othogonal eigenvectors and relative gaps
    I.S. Dhillon, B.N. Parlett
    SIAM J. Matrix Anal. Appl, 25(3), pp 858-899
  */

  /*
    int NegCount (SliceMatrix<> B, double mu)
    {
    double t = -mu;
    int negcnt = 0;
    int n = B.Width();
    for (int k = 0; k < n-1; k++)
    {
    double d = sqr(B(k,k)) + t;
    if (d < 0) negcnt++;
    t = t * sqr (B(k,k+1))/d - mu;
    }
    double d = sqr(B(n-1, n-1)) + t;
    if (d < 0) negcnt++;
    return negcnt;
    }
  */

  template <ORDERING ORD>
  int NegCount (SliceMatrix<double, ORD> B, double mu)
  {
    double t = -mu;
    int negcnt = 0;
    size_t n = B.Width();
    auto diag = B.Diag(0);
    auto super_diag = B.Diag(1);
    // cout << "Bptr = " << B.Data() << "diag.pr = " << diag.Data() << ", supe_diag.data = " << super_diag.Data() << endl;
    for (size_t k = 0; k < n-1; k++)
      {
        double d = sqr(diag(k)) + t;
        if (d < 0) negcnt++;
        // t = t * sqr (B(k,k+1))/d - mu;
        t = t * sqr (super_diag(k))/d - mu;
      }
    double d = sqr(diag(n-1)) + t;
    if (d < 0) negcnt++;
    return negcnt;
  }


  template <ORDERING ORD>
  SIMD<int64_t> NegCount (SliceMatrix<double, ORD> B, SIMD<double> mu)
  {
    SIMD<double> t = -mu;
    SIMD<int64_t> negcnt = 0;
    size_t n = B.Width();
    auto diag = B.Diag(0);
    auto super_diag = B.Diag(1);

    for (size_t k = 0; k < n-1; k++)
      {
        SIMD<double> d = sqr(diag(k)) + t;
        negcnt += If (d < SIMD<double>(0.0), SIMD<int64_t>(1), SIMD<int64_t>(0));
        t = t * sqr (super_diag(k))/d - mu;
      }
    SIMD<double> d = sqr(diag(n-1)) + t;
    negcnt += If (d < SIMD<double>(0.0), SIMD<int64_t>(1), SIMD<int64_t>(0));
    return negcnt;
  }

  // same as NegCount, using Golub-Kahan matrix
  int NegCountGK (FlatVector<> GK, double mu)
  {
    double t = -mu;
    int64_t negcnt = 0;
    size_t n = (GK.Size() + 1) / 2;

    for (size_t k = 0; k < n-1; k++)
      {
        double d = sqr(GK(2*k)) + t;

        if (d < 0)
        {
          negcnt++;
        }

        t = t * sqr (GK(2*k+1))/d - mu;
      }
    double d = sqr(GK(2*n-2)) + t;
    if (d < 0)
    {
      negcnt++;
    }

    return negcnt;
  }

  // same as NegCount, using Golub-Kahan matrix
  SIMD<int64_t> NegCountGK (FlatVector<> GK, SIMD<double> mu)
  {
    static Timer tgk("NegCountGK");
    RegionTimer reg(tgk);

    SIMD<double> t = -mu;
    SIMD<int64_t> negcnt = 0;
    size_t n = (GK.Size() + 1) / 2;

    for (size_t k = 0; k < n-1; k++)
      {
        SIMD<double> d = sqr(GK(2*k)) + t;
        negcnt += If (d < SIMD<double>(0.0), SIMD<int64_t>(1), SIMD<int64_t>(0));
        t = t * sqr (GK(2*k+1))/d - mu;
      }
    SIMD<double> d = sqr(GK(2*n-2)) + t;
    negcnt += If (d < SIMD<double>(0.0), SIMD<int64_t>(1), SIMD<int64_t>(0));

    return negcnt;
  }

  // same as NegCount, using Golub-Kahan matrix
  SIMD<int64_t> NegCountGK2 (FlatVector<> GK, SIMD<double> mu)
  {
    size_t n = GK.Size();

    SIMD<int64_t> negcnt = 0;
    SIMD<double> d = -mu;
    negcnt += If (d < SIMD<double>(0.0), SIMD<int64_t>(1), SIMD<int64_t>(0));
    for (int i=0; i<n; i++)
    {
      d = -mu - sqr(GK(i)) / d;
      negcnt += If (d < SIMD<double>(0.0), SIMD<int64_t>(1), SIMD<int64_t>(0));
    }

    return negcnt - (n+1)/2;
  }

  // returns biggest singular value of bidiagonal matrix B
  template <ORDERING ORD>
  double LargestSval (SliceMatrix<double,ORD> B)
  {
    size_t n = B.Width();

    double upperbound = 0;
    // int m = B.Height();
    for (int k = 0; k < B.Width(); k++)
      upperbound = max2(upperbound, fabs(B(k,k)));
    for (int k = 0; k < B.Width()-1; k++)
      upperbound = max2(upperbound, fabs(B(k+1,k)));
    upperbound *= 4; // * 3 may be sufficient

    double l = 0.0, u = upperbound;

    // calculate biggest eigenvalue
    for (int bisectit = 0; bisectit < 50; bisectit++)  // 1/2^53 < floating pnt prec
    {
        double mid = 0.5 * (u+l);
        int negcnt = NegCount (B, mid*mid);

        if (negcnt > n-1)
          u = mid;
        else
          l = mid;
    }

    double sigma0 = l;
    return sigma0;
  }

  // Calculate singular values of bidiagonal matrix B below eps*(biggest singular value) if number is below kmax
  // can be used for low-rank approximation
  template <ORDERING ORD>
  int CalcSValsBiDiagonalBinary (SliceMatrix<double,ORD> B, FlatVector<> singvals,
                          size_t kmax=-1, double eps=0)
  {

    int n = B.Width();

    // default: calculate all svals
    if (kmax < 0)
      kmax = n;

    static Timer tb("bisection"); RegionTimer reg(tb);

    double sigma1 = LargestSval(B);

    // number of singular vectors needed for low-rank approximation
    int num_svec;
    if (eps > 0)
      num_svec = n - NegCount (SliceMatrix(B), sqr(eps*sigma1));
    else
      num_svec = kmax;

    if (num_svec > kmax)
       return 0;

    // calculate remaining values
    for (size_t i = 0; i < num_svec; i += SIMD<double>::Size())
    {
      SIMD<double> x = 0;
      SIMD<int64_t>* xptr = (SIMD<int64_t>*) &x;
      SIMD<int64_t> evnr = SIMD<int64_t>(n-i-1)-SIMD<int64_t>::FirstInt(); // start with the largest
      for (int64_t b=(1UL << 62); b >= (1UL << 5); b >>= 1)
      {
        SIMD<double> tmp = 0;
        SIMD<int64_t>* tmpptr = (SIMD<int64_t>*) &tmp;
        *tmpptr = *xptr + SIMD<int64_t>(b);

        //        SIMD<int64_t> negcnt = NegCount (B, sqr(tmp));
        SIMD<int64_t> negcnt = NegCount (B, tmp);
        *xptr = If(negcnt > evnr, *xptr, *tmpptr);
      }
      x = sqrt(x);

      SIMD<mask64> mask(n-i);
      x.Store (&singvals[i], mask);
    }

    return num_svec;
  }


  // calculate eigenvalues below eval_limit
  // return number of calculated eigenvalues
  int CalcEvalsGKBinary (FlatVector<> GK, FlatVector<> evals, double eval_limit=-1)
  {
    static Timer tevals("MR3_svals");
    RegionTimer reg(tevals);

    int n = (GK.Size() + 1) / 2;

    // number of singular vectors needed for low-rank approximation
    int num_svec;
    if (eval_limit > 0)
      num_svec = n - NegCountGK (GK, sqr(eval_limit));
    else
      num_svec = n;

    // calculate remaining values
    for (size_t i = 0; i < num_svec; i += SIMD<double>::Size())
    {
      SIMD<double> x = 0;
      SIMD<int64_t>* xptr = (SIMD<int64_t>*) &x;
      SIMD<int64_t> evnr = SIMD<int64_t>(n-i-1)-SIMD<int64_t>::FirstInt(); // start with the largest

      // calculate one bit every iteration
      for (int64_t b=(1UL << 62); b >= (1UL << 8); b >>= 1)
      {
        SIMD<double> tmp = 0;
        SIMD<int64_t>* tmpptr = (SIMD<int64_t>*) &tmp;
        *tmpptr = *xptr + SIMD<int64_t>(b);

        SIMD<int64_t> negcnt = NegCountGK (GK, tmp);
        // SIMD<int64_t> negcnt = NegCountGK2 (GK, tmp);
        *xptr = If(negcnt > evnr, *xptr, *tmpptr);
      }
      x = sqrt(x);

      SIMD<mask64> mask(n-i);
      x.Store (&evals[i], mask);
    }

    return num_svec;
  }

  // Calculate singular values of bidiagonal matrix B below eps*(biggest singular value) if number is below kmax
  // can be used for low-rank approximation
  template <ORDERING ORD>
  int CalcSValsBiDiagonal (SliceMatrix<double,ORD> B, FlatVector<> singvals,
                          int kmax=-1, double eps=0)
  {
    int n = B.Width();

    // default: calculate all svals
    if (kmax < 0)
      kmax = n;

    static Timer tb("bisection"); RegionTimer reg(tb);

    double sigma1 = LargestSval(B);

    // number of singular vectors needed for low-rank approximation
    int num_svec;
    if (eps > 0)
      num_svec = n - NegCount (SliceMatrix(B), sqr(eps*sigma1));
    else
      num_svec = kmax;

    if (num_svec > kmax)
       return 0;

    // calculate remaining values
    for (size_t i = 0; i < num_svec; i += SIMD<double>::Size())
    {
      SIMD<double> l = 0.0, u = 2*sigma1;
      SIMD<int64_t> evnr = SIMD<int64_t>(n-i-1)-SIMD<int64_t>::FirstInt(); // start with the largest
      for (int bisectit = 0; bisectit < 50; bisectit++)  // 1/2^53 < floating pnt prec
      {
        SIMD<double> mid = 0.5 * (u+l);
        SIMD<int64_t> negcnt = NegCount (B, mid*mid);
        u = If(negcnt > evnr, mid, u);
        l = If(negcnt > evnr, l, mid);
      }
      SIMD<mask64> mask(kmax-i);
      l.Store (&singvals[i], mask);
    }

    return num_svec;
  }



  // stationary quotient-difference with shift
  // L D L^t - mu I  =  Lp Dp Lp^t
  void stqds (FlatVector<> D, FlatVector<> L, double mu,
              FlatVector<> Dp, FlatVector<> Lp)
  {
    size_t n = D.Size();
    Dp(0) = D(0)-mu;
    for (size_t i = 0; i < n-1; i++)
      {
        Lp(i) = D(i)*L(i) / Dp(i);
        Dp(i+1) = D(i)*L(i)*L(i) + D(i+1)-Lp(i)*D(i)*L(i) - mu;
      }
  }

  // differential form of stationary quotient-difference with shift
  // L D L^t - mu I  =  Lp Dp Lp^t
  void dstqds (FlatVector<> D, FlatVector<> L, double mu,
               FlatVector<> Dp, FlatVector<> Lp, FlatVector<> gamma)
  {
    static Timer tdstqds("MR3_dstqds");
    RegionTimer reg(tdstqds);

    size_t n = D.Size();
    double s = -mu;
    for (size_t i = 0; i < n-1; i++)
      {
        gamma(i) = s;
        Dp(i) = s+D(i);
        Lp(i) = D(i)*L(i)/Dp(i);
        s = Lp(i)*L(i)*s - mu;
      }
    Dp(n-1) = s + D(n-1);
    gamma(n-1) = s;
  }

  // differential form of stationary quotient-difference with shift
  // in-place version
  // L D L^t - mu I  =  Lp Dp Lp^t
  // store result in L, D
  void dstqds (FlatVector<> D, FlatVector<> L, double mu)
  {
    static Timer tdstqds("MR3_dstqds");
    RegionTimer reg(tdstqds);

    size_t n = D.Size();
    double s = -mu;
    double Dp, Lp;
    for (size_t i = 0; i < n-1; i++)
      {
        Dp = s+D(i);
        Lp = D(i)*L(i)/Dp;
        D(i) = Dp;

        s = Lp*L(i)*s - mu;
        L(i) = Lp;
      }
    D(n-1) = s + D(n-1);
  }

  // differential form of stationary quotient-difference with shift
  // L D L^t - mu I  =  Lp Dp Lp^t
  // ensures that 
  void dstqds2 (FlatVector<> D, FlatVector<> L, double mu,
               FlatVector<> Dp, FlatVector<> Lp, FlatVector<> gamma)
  {
    static Timer tdstqds("MR3_dstqds");
    RegionTimer reg(tdstqds);

    size_t n = D.Size();
    double s = -mu;
    double q;
    for (size_t i = 0; i < n-1; i++)
    {
      gamma(i) = s;
      Dp(i) = s+D(i);

      if (isinf(abs(s)) || isinf(abs(D(i))))
        q = 1;
      else
        q = s / Dp(i);

      Lp(i) = D(i)*L(i)/Dp(i);
      s = q * D(i) * sqr(L(i)) - mu;
    }
    Dp(n-1) = s + D(n-1);
    gamma(n-1) = s;
	}

  // differential form of stationary quotient-difference with shift
  // in-place version
  // L D L^t - mu I  =  Lp Dp Lp^t
  // store result in L, D
  void dstqds2 (FlatVector<> D, FlatVector<> L, double mu)
  {
    static Timer tdstqds("MR3_dstqds");
    RegionTimer reg(tdstqds);

    size_t n = D.Size();
    double s = -mu;
    double q;
    double Dp, Lp;
    for (size_t i = 0; i < n-1; i++)
    {
      Dp = s+D(i);

      if (isinf(abs(s)) || isinf(abs(D(i))))
        q = 1;
      else
        q = s / Dp;

      Lp = D(i)*L(i)/Dp;
      D(i) = Dp;

      s = Lp*L(i)*s - mu;
      L(i) = Lp;
    }
    D(n-1) = s + D(n-1);
  }



  // differential norm of the progressive qd transform
  // L D L^t - mu I  =  Um Dm Um^t
  void dqds (FlatVector<> D, FlatVector<> L, double mu,
             FlatVector<> Dm, FlatVector<> Um, FlatVector<> gamma)
  {
    size_t n = D.Size();
    double p = D(n-1) - mu;
    for (int i = n-2; i >= 0; i--)
      {
        gamma(i+1) += p + mu;
        Dm(i+1) = D(i)*L(i)*L(i) + p;
        double t = D(i)/Dm(i+1);
        Um(i) = L(i)*t;
        p = p*t-mu;
      }
    Dm(0) = p;
    gamma(0) += p+mu;
  }

  // differential norm of the progressive qd transform
  // L D L^t - mu I  =  Um Dm Um^t
  void dqds2 (FlatVector<> D, FlatVector<> L, double mu,
             FlatVector<> Dm, FlatVector<> Um, FlatVector<> gamma)
  {
    size_t n = D.Size();
    double p = D(n-1) - mu;
    double q;
    for (int i = n-2; i >= 0; i--)
      {
        gamma(i+1) += p + mu;
        Dm(i+1) = D(i)*sqr(L(i)) + p;
        if (isinf(abs(p)) || isinf(abs(Dm(i+1)))) {
          q = 1;
        }
        else {
          q = p / Dm(i+1);
        }
        double t = D(i)/Dm(i+1);
        Um(i) = L(i)*t;
        p = q*D(i)-mu;
      }
    Dm(0) = p;
    gamma(0) += p+mu;
  }

  // calculate singular values only
  template <ORDERING ORD>
  void TCalcSVD (SliceMatrix<double,ORD> A)
  {
      if (A.Width() > A.Height())
        {
          TCalcSVD (Trans(A));
          return;
        }

      // cout << "CalcSVD, h = " << A.Height() << ", w = " << A.Width() << endl;
      if (double norm = L2Norm(A); isnan(norm) || isinf(norm))
        {
          cout << "input matrix norm = " << norm << endl;
          cout << "mat = " << A << endl;
          throw Exception ("called SVD with nan-matrix");
        }

      static Timer t("CalcSVD"); RegionTimer reg(t);
      size_t m = A.Height();
      size_t n = A.Width();

      if (min(n,m) < 500) {
        ReduceBiDiagonal (A);
      }
      else {
        Matrix<double, ColMajor> Aband(n,n);
        ReduceBiDiagonalBlocked<ORD,ColMajor> (A, Aband);
        A.Diag(0) = SliceMatrix(Aband).Diag(0);
        A.Diag(1) = SliceMatrix(Aband).Diag(1);
      }

      Vector<> singvals(n);
      CalcSValsBiDiagonalBinary(A, singvals, n);

      A.Diag(0) = singvals;
      A.Diag(1) = 0;
  }

  // SVD without eigenvectors
  void CalcSVD (SliceMatrix<double,ColMajor> A)
  {
    TCalcSVD (A);
  }
  void CalcSVD (SliceMatrix<double,RowMajor> A)
  {
    TCalcSVD (A);
  }




  /* ***************** Bi-diagonal SVD by divide-and-conquer ************ */

  /*
    matrix is (n+1)*n
    alpha = m.Diag(0)
    beta = m.Diag(-1)
  */


  void CalcRecLapack (FlatVector<> alpha, FlatVector<> beta, FlatVector<> sigma,
                      SliceMatrix<double,ColMajor> U, SliceMatrix<double,ColMajor> V)
  {
    size_t n = alpha.Size();
    if (n == 0)
      {
        U(0,0) = 1;
        return;
      }

    if (U.Height() != n+1 || U.Width() != n+1) throw Exception ("U has wrong size");
    if (V.Height() != n || V.Width() != n) throw Exception ("V has wrong size");
    Matrix M(n+1, n);
    M = 0;
    M.Diag(0) = alpha;
    M.Diag(-1) = beta;
    Matrix Morig = M;
    Matrix<double,ColMajor> hU = U;
    cout << "lapack SVD of matrix " << endl << M << endl;
    LapackSVD (M, V, hU);
    U = Trans(hU);
    sigma = M.Diag(0);

    cout << "check lapack:" << endl;
    cout << "alpha, beta = " << alpha << ", " << beta << endl;
    cout << Truncate(Trans(U) * Morig * V) << endl;
    cout << "U = " << endl << U << endl;
    cout << "V = " << endl << V << endl;
  }





  class GivensRotation
  {
    double c, s;
    int i1, i2;
  public:
    GivensRotation (double ac, double as, int ai1, int ai2)
      : c(ac), s(as), i1(ai1), i2(ai2) { ; }

    void Apply (SliceVector<> v, bool trans = false)
    {
      double v1 = v(i1), v2 = v(i2);
      if (trans)
        {
          v(i1) = c * v1 + s * v2;
          v(i2) = -s * v1 + c * v2;
        }
      else
        {
          v(i1) = c * v1 - s * v2;
          v(i2) = s * v1 + c * v2;
        }
    }

    template <ORDERING OM>
    void Apply (SliceMatrix<double,OM> mat, bool trans = false)
    {
      for (int i = 0; i < mat.Width(); i++)
        Apply (mat.Col(i), trans);
    }
  };





  /*
    find solution within (lower, upper) of
    1 + \sum_i  vs(i)**2 / (lams(i)^2 - lam^2) = 0
  */
  // returns t-lower and upper-t
  // the smaller one is more accurate !
  tuple<double,double> SolveSecular (FlatVector<> lams,
                                     FlatVector<> vs, double lower, double upper)
  {
    // static Timer t("secular"); RegionTimer r(t);

    /*
      Rank-One Modification of the Symmetric Eigenproblem
      Bunch, Nielsen, Sorensen
      Numer Math 31, 31-48, 1978

      with modifications by Gu + Eisenstat for SVD

      "Solving Secular Equations Stably and Efficiently"
      Ren-Chang Li
      TR Report UCB/CSD-94-851, Berkeley, 1994
    */

    // cout << "solvesecular called, lower = " << lower << ", upper = " << upper << endl;
    ArrayMem<double,100> vpsi, vphi;
    ArrayMem<double,100> lampsi_shift, lamphi_shift;

    double lower2 = sqr(lower);
    double upper2 = sqr(upper);
    double mid2 = 0.5 * (upper2+lower2);

    for (size_t i = 0; i < lams.Size(); i++)
      {
        if (sqr(lams[i]) < mid2)
          {
            vpsi.Append(sqr(vs[i]));
            lampsi_shift.Append ((lams[i]-lower)*(lams[i]+lower));  // for roundoff !!
            // lampsi_shift.Append (sqr(lams[i])-lower2);
          }
        else
          {
            vphi.Append(sqr(vs[i]));
            // lamphi_shift.Append (sqr(lams[i])-upper2);
            lamphi_shift.Append ((lams[i]-upper)*(lams[i]+upper)); // for roundoff !!
          }
      }

    // RegionTimer rl(tl);

    double delta2 = upper2-lower2;


    int maxits = 100;

    double t2, Delta2;
    // starting value
    if (vphi.Size() == 0)
      {
        t2 = 0;
        for (int j = 0; j < vpsi.Size(); j++)
          t2 = max(t2, lampsi_shift[j] + vpsi[j]);
        t2 = min(delta2/2, 0.99*t2);
        Delta2 = delta2 - t2;
      }
    else
      // only for the middle-way
      t2 = Delta2 = delta2/2;


    for (int i = 0; i < maxits; i++)
      {
        double phi = 0, psi = 0, phiprime = 0, psiprime = 0;

        for (size_t j = 0; j < vpsi.Size(); j++)
          {
            double num = 1.0 / (lampsi_shift[j]-t2);
            psi += vpsi[j] * num;
            psiprime += vpsi[j] * sqr(num);
          }

        for (size_t j = 0; j < vphi.Size(); j++)
          {
            double num = 1.0 / (lamphi_shift[j]+Delta2);
            phi += vphi[j] * num;
            phiprime += vphi[j] * sqr(num);
          }


        double w = 1 + phi + psi;
        // cout << "it = " << i << ", w = " << w << " npsi = " << vpsi.Size() << ", nphi = " << vphi.Size() << " t2 = " << t2 << " Detela2 = " << Delta2 << endl;

        if (vphi.Size() > 0)
          {
            double a = (Delta2-t2) * w + Delta2*t2*(phiprime+psiprime);
            double b = -Delta2*t2*w;
            double c = w+t2*psiprime-Delta2*phiprime;
            // double discr = a*a-4*b*c;
            // cout << "discr = " << discr << endl;
            // double discr = sqr(delta2*w) + 2 * w * Delta2*t2*delta2*(psiprime-phiprime) + sqr (Delta2*t2*(phiprime+psiprime));

            double discr = sqr ( delta2*w+Delta2*t2*(psiprime-phiprime) ) + 4*sqr(t2*Delta2)*psiprime*phiprime;
            if (discr < 0) discr = 0;
            double incr = (a > 0) ? 2*b/(a+sqrt(discr)) : (a-sqrt(discr)) / (2*c);

            if (Delta2-incr < 1e-6 * Delta2) incr *= 0.999999;
            if (t2+incr < 1e-6 * t2) incr *= 0.999999;

            t2 += incr;
            Delta2 -= incr;
          }
        else
          {
            double incr = (1+psi)/psiprime*psi;
            t2 += incr;
            Delta2 -= incr;
          }

        if (fabs (w) < 1e-14 * (1.0+fabs(phi)+fabs(psi)))
          if (i < maxits-2) i = maxits-2;

        // if (i > 20 && i < 20)
        // cout << "many its, i = " << i << endl;

        if (t2 < Delta2)
          Delta2 = delta2 - t2;
        else
          t2 = delta2 - Delta2;



        if (vphi.Size() > 0)
          if (i > 60 && i < maxits-5)
            {
              cout << "i = " << i << ", w = " << w << ", phi = " << phi << ", psi = " << psi
                   << ", t2 = " << t2 << ", Delta2 = " << Delta2 << endl;
              // << " inc = " << incr << endl;
              if (i == maxits-7)
                {
                  cout << "lower/upper = " << lower << "/" << upper << endl;
                  cout << "vpsi = " << vpsi << endl;
                  cout << "lampsi_shift = " << lampsi_shift << endl;
                  cout << "vphi = " << vphi << endl;
                  cout << "lamphi_shift = " << lamphi_shift << endl;
                  throw Exception ("too many its");
                }
            }
      }



    if (isnan(t2) || isinf(t2)) throw Exception ("t2 is nan");
    if (isnan(Delta2) || isinf(Delta2)) throw Exception ("t2 is nan");

    if (t2 < 0)
      {
        cout << "lower/upper = " << lower << "/" << upper << endl;
        cout << "t2/Delta2 = " << t2 << "/" << Delta2 << endl;
        // cout << "phip1_inv = " << phip1_inv << endl;
        cout << "lams = " << lams << endl;
        cout << "vs = " << vs << endl;
        cout << "vpsi = " << vpsi << endl;
        cout << "lampsi_shift = " << lampsi_shift << endl;
        cout << "vphi = " << vphi << endl;
        cout << "lamphi_shift = " << lamphi_shift << endl;
        throw Exception ("t2 negative");
      }
    if (Delta2 < 0 && vphi.Size() > 0)
      {
        cout << "lams = " << lams << endl;
        cout << "vs = " << vs << endl;
        throw Exception ("Delta2 negative");
      }

    return { t2, Delta2 };
  }


  void CalcSVDArrowHead (FlatVector<> z, FlatVector<> D,
                         FlatVector<> sigma,
                         SliceMatrix<double,ColMajor> MU, SliceMatrix<double, ColMajor> MV,
                         FlatMatrix<double,ColMajor> Q, FlatMatrix<double,ColMajor> W,
                         SliceMatrix<double,ColMajor> U, SliceMatrix<double,ColMajor> V)
  {
    static Timer t("SVD - ArrowHead"); RegionTimer reg(t);
    size_t n = z.Size();
    double normmat = L2Norm(z) + L2Norm(D) + 1e-40;

    ArrayMem<int,100> index(n);
    for (int i : Range(n)) index[i] = i;
    QuickSortI (FlatArray(D.Size(), D.Data()), index.Range(1,n));

    VectorMem<100> zs(n), Ds(n);
    for (size_t i = 0; i < n; i++)
      {
        zs(i) = z[index[i]];
        Ds(i) = D[index[i]];
      }


    // deflation  (Gu + Eisenstat, Chap 4.1)
    // if Ds are the same, rotate up

    ArrayMem<bool,100> sameDs(n);
    sameDs = false;
    for (int i = n-2; i >= 0; i--)
      {
        if ( (Ds(i+1) - Ds(i)) < 1e-14*normmat)
          {
            sameDs[i+1] = true;
            double c = zs(i), s = -zs(i+1);
            double r = hypot (c, s);
            if (r > 1e-20 * normmat)
              {
                c /= r; s /= r;
                GivensRotation G(c, s, index[i], index[i+1]);
                zs(i) = r;
                zs(i+1) = 0;
                G.Apply (SliceMatrix(Trans(Q)));
                if (i > 0)
                  G.Apply (SliceMatrix(Trans(W)));
              }
            else
              {
                zs(i) = 0;
                zs(i+1) = 0;
              }
          }
      }
    for (size_t i = 1; i < n; i++)
      if (sameDs[i])
        Ds(i) = Ds(i-1);

    ArrayMem<bool,100> deflated(n);
    deflated = false;
    size_t num_deflated = 0;
    if (fabs(zs(0)) < 1e-14 * normmat) zs(0) = 1e-14 * normmat;
    for (int i = 1; i < n; i++)
      if (fabs(zs(i)) < 1e-14 * normmat)
        {
          zs(i) = 0;
          deflated[i] = true;
          num_deflated++;
        }


    ArrayMem<int,100> deflated_ind(num_deflated);
    ArrayMem<int,100> non_deflated_ind(n-num_deflated);

    size_t cnt_def = 0, cnt_nondef = 0;
    for (size_t i = 0; i < n; i++)
      {
        if (deflated[i])
          deflated_ind[cnt_def++] = i;
        else
          non_deflated_ind[cnt_nondef++] = i;
      }
    size_t ncomp = cnt_nondef;
    VectorMem<100,double> compress_Ds(ncomp+1);  // includes extra one at the end
    VectorMem<100,double> compress_zs(ncomp);

    for (size_t i : Range(ncomp))
      {
        int i_orig = non_deflated_ind[i];
        compress_Ds(i) = Ds(i_orig);
        compress_zs(i) = zs(i_orig);
      }

    compress_Ds(ncomp) = compress_Ds(ncomp-1)+L2Norm(compress_zs);

    VectorMem<100> compress_omega(n-num_deflated),
      compress_omega2diff(n-num_deflated),
      compress_omega2ref(n-num_deflated);

    // static Timer tsec("Solve Secular");
    // tsec.Start();

    for (size_t i = 0; i < ncomp; i++)
      {
        double l = compress_Ds(i);
        double u = compress_Ds(i+1);

        auto [t2,Delta2] = SolveSecular (compress_Ds.Range(ncomp), compress_zs, l, u);

        if (t2 < Delta2)
          {
            compress_omega2ref(i) = sqr(l);
            compress_omega2diff(i) = t2;
          }
        else
          {
            compress_omega2ref(i) = sqr(u);
            compress_omega2diff(i) = -Delta2;
          }
        compress_omega(i) = sqrt (compress_omega2ref(i)+compress_omega2diff(i));
      }
    // tsec.Stop();

    VectorMem<100,double> compress_Ds2(ncomp);
    for (size_t i = 0; i < ncomp; i++)
      compress_Ds2(i) = sqr(compress_Ds(i));


    VectorMem<100> compress_zsmod(ncomp);
    static Timer tzmod("zmod");
    tzmod.Start();

    VectorMem<100> shifted(ncomp);
    compress_zsmod = 1;
    for (size_t j = 0; j < ncomp-1; j++)
      {
        for (size_t i = 0; i < ncomp; i++)
          shifted(i) = compress_omega2ref(j)-compress_Ds2(i);   // roundoff is here critical !
        for (size_t i = 0; i <= j; i++)
          compress_zsmod(i) *= (shifted(i)+compress_omega2diff(j)) / (compress_Ds2(j+1)-compress_Ds2(i));
        for (size_t i = j+1; i < ncomp; i++)
          compress_zsmod(i) *= (shifted(i)+compress_omega2diff(j)) / (compress_Ds2(j)-compress_Ds2(i));
      }

    for (size_t i = 0; i < ncomp; i++)
      shifted(i) = compress_omega2ref(ncomp-1)-compress_Ds2(i);
    for (size_t i = 0; i < ncomp; i++)
      compress_zsmod(i) *= shifted(i)+compress_omega2diff(ncomp-1);

    for (auto & zsmod : compress_zsmod)
      zsmod = sqrt(zsmod);

    tzmod.Stop();

    for (size_t i = 0; i < compress_zs.Size(); i++)
      if (compress_zs(i) < 0) compress_zsmod(i)*=-1;

    for (size_t i = 0; i < num_deflated; i++)
      {
        sigma(ncomp+i) = Ds(deflated_ind[i]);
        U.Col(ncomp+i) = Q.Col(index[deflated_ind[i]]);
        V.Col(ncomp+i) = W.Col(index[deflated_ind[i]]);
      }

    // reorder Q,W, using mem from U,V
    for (size_t i = 0; i < ncomp; i++)
      {
        U.Col(i) = Q.Col(index[non_deflated_ind[i]]);
        V.Col(i) = W.Col(index[non_deflated_ind[i]]);
      }
    Q.Cols(ncomp) = U.Cols(ncomp);
    W.Cols(ncomp) = V.Cols(ncomp);

    static Timer tuv("tuv");
    tuv.Start();
    for (size_t i = 0; i < ncomp; i++)
      {
        auto colu = MU.Col(i).Range(ncomp);
        auto colv = MV.Col(i).Range(ncomp);
        sigma(i) = compress_omega(i);

        for (size_t j = 0; j < ncomp; j++)
          shifted(j) = compress_Ds2(j)-compress_omega2ref(i); // critical for roundoff

        for (size_t j = 0; j < ncomp; j++)
          {
            double ui = compress_zsmod(j) / (shifted(j)-compress_omega2diff(i));
            colu(j) = ui;
            colv(j) = compress_Ds(j)*ui;
          }
        colv(0) = -1;

        colu /= L2Norm(colu);
        colv /= L2Norm(colv);
      }
    tuv.Stop();


    static Timer tmult("multQU");
    tmult.AddFlops(2*Q.Height()*ncomp*ncomp);
    tmult.Start();
    U.Cols(ncomp) = Q.Cols(ncomp) * MU.Cols(ncomp).Rows(ncomp);
    V.Cols(ncomp) = W.Cols(ncomp) * MV.Cols(ncomp).Rows(ncomp);
    tmult.Stop();
    /*
      // testing
    if (double erru = L2Norm(Matrix(Trans(MU)*MU)-Identity(n)); erru > 1e-12)
      cout << "Secular, orthogonal U error = " << erru << endl;
    if (double errv = L2Norm(Matrix(Trans(MV)*MV)-Identity(n)); errv > 1e-12)
      cout << "Secular, orthogonal V error = " << errv << endl;
    */


    // cout << "Ds2 = " << Ds2 << endl;
    // cout << "omega = " << omega << endl;
    /*
    if (isnan(L2Norm(omega)) || isinf(L2Norm(omega)))
      {
        cout << "omega is illegal" << endl;
        cout << "normmat = " << normmat << endl;
        cout << "omega = " << endl << omega;
        cout << "zs = " << endl << zs << endl;
        cout << "D^2s = " << endl << Ds2 << endl;
        cout << "D_next = " << endl << Ds_next << endl;
        cout << "deflated = " << endl << deflated << endl;
      }
    */

    /*
    for (size_t i = 0; i < n; i++)
      sigma(i) = omega(i);
    */
  }



  /*
    Matrix is (n+1) * n
    alpha diagonal, beta .. sub-diagonal
    sigma(n) singular values
    U   (n+1) * (n+1)  .... first n cols are singular vectors
    V   n * n


    first n columns form the reduced SVD
    if alpha(0) == 0, then first row of first cols is 0
    i.e. U(1:n+1, 0:n) is U-factor of reduced matrix
  */
  void CalcSVDBiDiagonalRec (FlatVector<> alpha, FlatVector<> beta, FlatVector<> sigma,
                             SliceMatrix<double,ColMajor> U, SliceMatrix<double,ColMajor> V)
  {
    /*
    static Timer t("CalcSVDBiDiagonalRec"); RegionTimer r(t);
    static Timer t1("CalcSVDBiDiagonalRec 1");
    static Timer t2("CalcSVDBiDiagonalRec 2");
    static Timer t2b("CalcSVDBiDiagonalRec 2b");
    static Timer tWQ("CalcSVDBiDiagonalRec Build WQ");
    static Timer tzD("CalcSVDBiDiagonalRec Build zD");
    static Timer tdef("deflation");
    static Timer tMUV("MU,MV");
    static Timer tsort("sort");
    */


    /*
      CalcRecLapack (alpha, beta, sigma, U, V);
      cout << "lapack sigma = " << sigma << endl;
      return;
    */

    // cout << "CalcSVDBiDiagonalRec called, n = " << alpha.Size() << endl;
    // cout << "CalcSVDBiDiagonalRec, alpha " << endl << alpha << endl << "beta = " << endl << beta << endl;

    size_t n = alpha.Size();
    if (n == 0)
      {
        U(0,0) = 1;
        return;
      }

    if (n == 1)
      {
        double r = sqrt(sqr(alpha(0))+sqr(beta(0)));
        sigma(0) = r;
        if (r == 0)
          {
            U(0,0) = 0; U(0,1) = 1;
            U(1,0) = 1; U(1,1) = 0;
            V(0,0) = 1;
          }
        else
          {
            double c = alpha(0) / r, s = beta(0) / r;
            U(0,0) = c;
            U(1,0) = s;
            U(0,1) = -s;
            U(1,1) = c;
            V(0,0) = 1;
          }
        return;
      }

    // t1.Start();

    double maxval = 0;
    for (size_t i = 0; i < n; i++)
      {
        maxval = max(maxval, fabs(alpha(i)));
        maxval = max(maxval, fabs(beta(i)));
      }

    if (maxval == 0.0)
      {
        sigma = 0.0;
        V = 0.0;
        V.Diag(0) = 1;
        U = 0.0;
        U.Rows(1,n+1).Cols(0,n).Diag(0) = 1;
        U(0,n) = 1;
        return;
      }


    double scale = (maxval != 0) ? 1/maxval : 0;
    VectorMem<100,double> alpha_scaled(n), beta_scaled(n);
    alpha_scaled = scale * alpha;
    beta_scaled = scale * beta;


    // double normmat = L2Norm(alpha_scaled) + L2Norm(beta_scaled) + 1e-40;

    /*
    Matrix<> B(n+1, n);  // for testing
    B = 0;
    B.Diag(0) = alpha;
    B.Diag(-1) = beta;
    */

    size_t k = n/2;
    double alphak = alpha_scaled(k);
    double betak = beta_scaled(k);

    // borrow memory ...
    auto U1 = U.Rows(k+1).Cols(k+1);
    auto U2 = U.Rows(k+1,n+1).Cols(k+1,n+1);
    auto W1 = V.Rows(k).Cols(k);
    auto W2 = V.Rows(k,n-1).Cols(k,n-1);

    VectorMem<100> D(n);
    D(0) = 0;

    // t1.Stop();
    CalcSVDBiDiagonalRec (alpha_scaled.Range(0,k), beta_scaled.Range(0,k), D.Range(1,k+1), U1, W1);
    CalcSVDBiDiagonalRec (alpha_scaled.Range(k+1,n), beta_scaled.Range(k+1,n), D.Range(k+1,n), U2, W2);

    // tWQ.Start();

    /*
    if (isnan(L2Norm(D)) || isinf(L2Norm(D)))
      {
        cout << "D has nans" << endl;
        cout << "alpha = " << alpha << endl;
        cout << "beta = " << beta << endl;
        cout << "D = " << D << endl;
        cout << "U1 = " << U1 << endl;
        cout << "W1 = " << W1 << endl;
        cout << "U2 = " << U2 << endl;
        cout << "W2 = " << W2 << endl;
        throw Exception("D has nans");
      }
    */

    /*
      cout << "own recursive: D = " << D << endl;
      cout << "U1,V1 = " << endl << U1 << endl << W1 << endl;
      cout << "U2,V2 = " << endl << U2 << endl << W2 << endl;
    */
    // CalcRecLapack (alpha.Range(0,k), beta.Range(0,k), D.Range(1,k+1), U1, W1);
    // CalcRecLapack (alpha.Range(k+1,n), beta.Range(k+1,n), D.Range(k+1,n), U2, W2);
    /*
    cout << "Lapack recursive: D = " << D << endl;
    cout << "U1,V1 = " << endl << U1 << endl << W1 << endl;
    cout << "U2,V2 = " << endl << U2 << endl << W2 << endl;
    */


    auto Q1 = U1.Cols(0,k);
    auto q1 = U1.Col(k);
    auto Q2 = U2.Cols(0,n-k-1);
    auto q2 = U2.Col(n-k-1);

    double lam1 = q1(k);
    auto l1 = Q1.Row(k);
    double phi2 = q2(0);
    auto f2 = Q2.Row(0);

    double r0 = hypot(alphak*lam1, betak*phi2);

    double c0, s0;
    if (r0 != 0)
      { c0 = alphak*lam1/r0; s0 = betak*phi2/r0; }
    else
      { c0 = 0; s0 = 1; }


    // Matrix<> Q(n+1,n), W(n,n);
    ArrayMem<double,512> memQW((2*n+1)*n);
    FlatMatrix<double,ColMajor> Q(n+1, n, memQW.Data());
    FlatMatrix<double,ColMajor> W(n, n, memQW.Data()+(n+1)*n);

    Q = 0.0;
    auto [Qtop,Qbot] = Q.SplitRows(k+1);
    Qtop.Col(0) = c0*q1;
    Qbot.Col(0) = s0*q2;
    Qtop.Cols(1,1+k) = Q1;
    Qbot.Cols(1+k,n) = Q2;
    W = 0.0;
    W.Rows(0,k).Cols(1,k+1) = W1;
    W(k,0) = 1;
    W.Rows(k+1,n).Cols(k+1,n) = W2;
    VectorMem<100> q(n+1);
    q.Range(0,k+1) = -s0*q1;
    q.Range(k+1,n+1) = c0*q2;

    // now, B = Q * M * Trans(W)
    // with M = diag + first col
    /*
      cout << "arrow head before deflation" << endl;
      Matrix M = Matrix(Trans(Q)*B) * W;
      cout << "Q^T B W = " << Truncate(M) << endl;
      cout << "q^T B = " << Truncate( Trans(B)*q ) << endl;
    */

    // tWQ.Stop();
    // tzD.Start();

    VectorMem<100> z(n);
    z(0) = r0;
    z.Range(1,k+1) = alphak*l1;
    z.Range(k+1,n) = betak*f2;

    // Matrix<double,ColMajor> MU(n,n), MV(n,n);   // M = MU * Msigma * MV
    ArrayMem<double,512> memMUV(2*n*n);
    FlatMatrix<double,ColMajor> MU(n,n,memMUV.Data());
    FlatMatrix<double,ColMajor> MV(n,n,memMUV.Data()+n*n);

    CalcSVDArrowHead (z, D, sigma, MU, MV, Q, W, U.Cols(n), V);

    // tMUV.Stop();
    // tsort.Start();

    // U.Cols(n) = Q * MU;
    U.Col(n) = q;
    // V = W * MV;



    // we still need sorting, deflated and non-deflated may overlap
    ArrayMem<int,100> index(n);
    for (int i : Range(n)) index[i] = i;
    QuickSortI (FlatArray(sigma.Size(), sigma.Data()), index, [](double x, double y) { return y < x; });

    VectorMem<100> tmp_sigma(n);
    tmp_sigma = sigma;
    auto tmp_MU = Q.Cols(n);  // reuse memory
    auto tmp_MV = W.Cols(n);

    static Timer tcopyuv("svd - copyuv");
    static Timer tsortuv("svd - sortuv");

    tcopyuv.Start();
    tmp_MU = U.Cols(n);
    tmp_MV = V.Cols(n);
    tcopyuv.Stop();
    tsortuv.Start();
    for (size_t i = 0; i < n; i++)
      {
        sigma(i) = tmp_sigma(index[i]);
        U.Col(i) = tmp_MU.Col(index[i]);
        V.Col(i) = tmp_MV.Col(index[i]);
      }
    tsortuv.Stop();

    sigma *= maxval;


    // tsort.Stop();
    /*
      Matrix Sigma(n);
      Sigma = 0.0; Sigma.Diag() = sigma;
      cout << "shold be same arrow-head:" << endl
      << Truncate( Matrix ( Matrix(MU*Sigma) * Trans(MV) ) ) << endl;
    */

    // static Timer tmatmult("matmult");
    // tmatmult.Start();
    // tmatmult.Stop();

    /*
    if (isnan (L2Norm(sigma)))
      throw Exception ("sigma is none");
    */

      // testing

    /*
    double errMU = L2Norm(Trans(MU)*MU-Identity(n));
    double errMV = L2Norm(Trans(MV)*MV-Identity(n));

    if (max(errMU, errMV) > 1e-10)
      {
        cout << "MU orthogonality err = " << errMU << endl;
        cout << "MV orthogonality err = " << errMV << endl;

        if (max(errMU, errMV) > 1e-8)
          {
            cout << "MU = " << endl << MU << endl;
            cout << "MV = " << endl << MV << endl;
            cout << "Trans(MU)*MU-I = " << endl << Truncate(Trans(MU)*MU-Identity(n)) << endl;
            cout << "Trans(MV)*MV-I = " << endl << Truncate(Trans(MV)*MV-Identity(n)) << endl;
            cout << "alpha = " << endl << alpha << endl;
            cout << "beta = " << endl << beta << endl;
            // cout << "omega2 = " << omega2 << endl;
            // cout << "omega2ref = " << omega2ref << endl;
            // cout << "omega2diff = " << omega2diff << endl;
            cout << "Ds = " << Ds << endl;
            cout << "zs = " << zs << endl;
            cout << "omega = " << omega << endl;
            cout << "degen = " << deflated << endl;
            if (max(errMU,errMV) > 1e-3)
              throw Exception ("bad orthogonality");
          }
      }
    */



    /*
      // testing
    Matrix Sigma = Matrix(Trans(U)*B)*V;
    Sigma.Diag(0) = 0.0;
    if (double errsigma = L2Norm(Sigma); errsigma > 1e-4)
      {
        cout << "in DnC, bidiag, sigma err = " << errsigma << endl;
        if (Sigma.Height() <= 6)
          {
            cout << "Sigma clear diag = " << endl << Sigma << endl;
            cout << "Sigma clear diag = " << endl << Truncate(Sigma, 1e-10) << endl;
            Sigma = Matrix(Trans(U)*B)*V;
            cout << "bidiag, Sigma = " << endl << Truncate(Sigma) << endl;
            cout << "alpha = " << endl << alpha << endl;
            cout << "beta = " << endl << beta << endl;
            cout << "deflation = " << endl << deflated << endl;
            cout << "zs = " << endl << zs << endl;
            cout << "zsmod = " << endl << zsmod << endl;
            cout << "Ds = " << endl << Ds << endl;
          }
        // throw Exception("wrong");
      }
    */
  }

  /*
    B ... n*n,   uppper bi-diagonal
    Compute   B = U Sigma V^T
    with Sigma stored as diag B
  */
  template <ORDERING ORD>
  void CalcSVDBiDiagonal (SliceMatrix<double,ORD> B,
                          SliceMatrix<double,ColMajor> U, SliceMatrix<double,ColMajor> V)
  {
    static Timer t("CalcSVDBiDiagonal"); RegionTimer r(t);

    // Matrix<> Borig = B; // for testing
    size_t n = B.Height();
    VectorMem<100> alpha(n), beta(n), sigma(n);

    alpha(0) = 0;
    alpha.Range(1,n) = B.Diag(1);
    beta = B.Diag(0);

    Matrix<double,ColMajor> U1(n+1, n+1);
    Matrix<double,ColMajor> V1(n,n);

    CalcSVDBiDiagonalRec (alpha, beta, sigma, U1, V1);

    // in case of LowRankApproximation k < n (we don't need all singular vectors)
    size_t k = U.Width();

    U = U1.Rows(1,n+1).Cols(0, k);
    V = V1.Cols(0,k);

    B = 0.0;
    B.Diag(0) = sigma;


    /*
      // testing
    Matrix Sigma = Matrix(Trans(U)*Borig)*V;
    Sigma.Diag(0) = 0.0;
    if (double errsigma = L2Norm(Sigma); errsigma > 1e-4)
      {
        cout << "bidiag, sigma err = " << errsigma << endl;
        if (Sigma.Height() <= 6)
          {
            cout << "Sigma clear diag = " << endl << Sigma << endl;
            cout << "Sigma clear diag = " << endl << Truncate(Sigma, 1e-10) << endl;
            Sigma = Matrix(Trans(U)*Borig)*V;
            cout << "bidiag, Sigma = " << endl << Truncate(Sigma) << endl;
            cout << "alpha = " << endl << alpha << endl;
            cout << "beta = " << endl << beta << endl;
          }
        // throw Exception("wrong");
      }
    */
  }



  // A = U * Sigma * V^T
  template <ORDERING ORD>
  void TCalcSVD (SliceMatrix<double,ORD> A,
                 SliceMatrix<double, ColMajor> U, SliceMatrix<double, ColMajor> V)
  {
    /*
    Matrix Aorig = A;
    Matrix<double,ColMajor> hU = U;
    LapackSVD (A, V, hU);
    U = Trans (hU);
    // cout << "have svd, sigma = " << SliceMatrix(A).Diag(0) << endl;
    // cout << "Trans(U) * A * V = " << Truncate(Trans(U) * Aorig * V) << endl;
    return;
    */

    if (A.Width() > A.Height())
      {
        TCalcSVD (Trans(A), V, U);
        return;
      }

    // cout << "CalcSVD, h = " << A.Height() << ", w = " << A.Width() << endl;
    if (double norm = L2Norm(A); isnan(norm) || isinf(norm))
      {
        cout << "input matrix norm = " << norm << endl;
        cout << "mat = " << A << endl;
        throw Exception ("called SVD with nan-matrix");
      }

    static Timer t("CalcSVD"); RegionTimer reg(t);
    size_t m = A.Height();
    size_t n = A.Width();

    // Matrix<> A_orig = A;   // for testing

    Matrix<double, ColMajor> U1(m);
    Matrix<double, ColMajor> V1(n);

    if (min(n,m) < 500)
      ReduceBiDiagonal (A, U1, V1);
    else
      ReduceBiDiagonalBlocked<ORD,ColMajor> (A, U1, V1);

    // testting
    /*
      if (m < 50)
      {
      Matrix B = Matrix(Trans(U1) * A_orig) * V1;   // bi-diagonal
      cout << "U^T * A V" << endl << Truncate(B) << endl;
      }
    */

    Matrix<double,ColMajor> UB(m), VB(n);
    UB = Identity(m);
    // cout << "diag A = " << endl << A.Diag(0) << endl;
    // cout << "super-diag A = " << endl << A.Diag(1) << endl;
    CalcSVDBiDiagonal (A.Rows(n), UB.Rows(n).Cols(n), VB);

    {
      static Timer tmultUV("CalcSVD, mult U1*UB, V1*VB"); RegionTimer regUV(tmultUV);
      tmultUV.AddFlops (n*n*n+m*m*m);
      U = U1*UB;
      V = V1*VB;
    }

    /*
       // testing ...
    double errU = L2Norm (U * Trans(U) - Identity(U.Height()));
    double errV = L2Norm (V * Trans(V) - Identity(V.Height()));
    if (errU + errV > 1e-2)
      {
        cout << "orthogonaliry error = " << errU + errV << endl;
        // cout << "Aorig = " << A_orig << endl;
        // cout << "diag = " << A.Diag(0) << endl;
        // cout << "superd = " << A.Diag(1) << endl;
        cout << "err ortho U: " << errU << endl;
        cout << "err ortho V: " << errV << endl;
        cout << "err ortho U1: " << L2Norm (U1 * Trans(U1) - Identity(U1.Height())) << endl;
        cout << "err ortho V1: " << L2Norm (V1 * Trans(V1) - Identity(V1.Height())) << endl;
        cout << "err ortho UB: " << L2Norm (UB * Trans(UB) - Identity(UB.Height())) << endl;
        cout << "err ortho VB: " << L2Norm (VB * Trans(VB) - Identity(VB.Height())) << endl;
        // cout << "UB = " << endl << UB << endl;
        // cout << "VB = " << endl << VB << endl;
        if (errU+errV > 1e-4)
          throw Exception ("got a bad SVD");
      }
    */

    /*
    // testing ...
    Matrix Sigma = Matrix(Trans(U)*A_orig)*V;
    Sigma.Diag(0) = 0.0;
    if (double errsigma = L2Norm(Sigma); errsigma > 1e-2)
      {
        cout << "sigma err = " << errsigma << endl;
        Sigma = Matrix(Trans(U)*A_orig)*V;
        cout << "Sigma = " << endl << Truncate(Sigma) << endl;
        cout << "A_orig = " << endl << A_orig << endl;
        cout << "diag = " << endl << A.Diag(0) << endl;
        cout << "super-diag = " << endl << A.Diag(1) << endl;
        if (errsigma > 1e-2)
          throw Exception("Sigma not diagonal");
      }
    */
    // cout << "sigma = " << A.Diag(0) << endl;
  }

  void CalcSVD (SliceMatrix<double,RowMajor> A,
                SliceMatrix<double, ColMajor> U, SliceMatrix<double, ColMajor> V)
  {
    TCalcSVD (A, U, V);
  }
  void CalcSVD (SliceMatrix<double,ColMajor> A,
                SliceMatrix<double, ColMajor> U, SliceMatrix<double, ColMajor> V)
  {
    TCalcSVD (A, U, V);
  }


  /************* MRRR - Multiple Realtively Robust Representations ****************/

  // needed for MR3
  double reldist2(double x, double y)
  {
    return fabs(x - y) / fabs(x);
  }


  // uses twisted factorization to calculate eigenvector to approximated eigenvalue
  double getvec (FlatVector<> L, FlatVector<> D,
               double eval, FlatVector<> z, bool normed=1)
  {
    size_t n = D.Size();

    Vector Dp(n), Lp(n-1);
    Vector Dm(n), Um(n-1);
    Vector gamma(n);

    // LDL^T - lambda = Lp Dp Lp^T
    dstqds2 (D, L, eval, Dp, Lp, gamma);

    // LDL^T - lambda = Um Dm Um^T
    dqds2 (D, L, eval, Dm, Um, gamma);

    //tcalcvec.Start();
    // find twist index
    int r = 0;
    for (int i = 0; i < n; i++)
      if (fabs(gamma(i)) < fabs(gamma(r)))
        r = i;

    // calculate Vector from twisted factorization
    z(r) = 1;
    for (int i = r-1; i >= 0; i--)
    {
      if (Dp(i) != 0)
        z(i) = -Lp(i)*z(i+1);
      else
        z(i) = -(D(i+1)*L(i+1)/(D(i)*L(i))) * z(i+2);
    }
    for (int i = r; i < n-1; i++)
    {
      if (Dm(i+1) != 0)
        z(i+1) = -Um(i)*z(i);
      else
        z(i+1) = -(D(i-1)*L(i-1)/(D(i)*L(i))) * z(i-1);
    }

    if (normed)
      z /= L2Norm(z);

    return gamma(r);
  }

  void RQI(FlatVector<> L, FlatVector<> D, double lam, FlatVector<> z)
  {
    static Timer trqi("MR3_rqi");
    RegionTimer reg(trqi);

    size_t count = 0;

    double gamma = getvec(L, D, lam, z, 0);
    double last_residual = fabs(gamma) / L2Norm(z);
    double residual;
    do {
      double gamma = getvec(L, D, lam, z, 0);
      count++;

      double znorm = L2Norm(z);
      residual = fabs(gamma) / znorm;
      if (last_residual <= residual && count > 1 || count > 4) {
        break;
      }

      lam += fabs(gamma) / sqr(znorm);

      last_residual = residual;

    } while (1);
  //  z /= L2Norm(z);
  }

  // T_GK = LDL^T factorization of Golub-Kahan matrix
  // store as singular vectors u and v
  // TODO: remove position?
  void getvecGK(FlatVector<> L, FlatVector<> D, double eval,
                  FlatVector<> u, FlatVector<> v, int position=0, bool rqi=0)
  {
    static Timer tgetvec("MR3_getvec");
    RegionTimer reg(tgetvec);

    size_t n = D.Size();

    Vector z(n);

    // RQI works better for recursive calls (in CalcSVecsMR3GK)
    if (rqi) {
      RQI(L, D, eval, z);
    }
    else {
      getvec(L, D, eval, z, 0);
    }

    //tfillvu.Start();
    for (int i = 0; i < n / 2; i++) {
      u(i) = z(2*i+1);
      v(i) = z(2*i);
    }
    u /= L2Norm(u);
    v /= L2Norm(v);
    //tfillvu.Stop();

  }


  // Calculate SVD from bidiagonal matrix B using twisted factorization (RRR)
  // predecessor to MR3
  void CalcSVDBiDiagonal1 (SliceMatrix<> B,
                           FlatMatrix<double,ColMajor> U, FlatMatrix<double,ColMajor> V)
  {
    static Timer tb("tbisect");
    static Timer tbi("tsolvebi");
    static Timer tub("tsmallUV");
    double upperbound = 0;

    size_t n = B.Width();
    Vector<> singvals(n);

    /*
    // int m = B.Height();
    for (int k = 0; k < B.Width(); k++)
      upperbound = max2(upperbound, fabs(B(k,k)));
    for (int k = 0; k < B.Width()-1; k++)
      upperbound = max2(upperbound, fabs(B(k+1,k)));
    upperbound *= 3; // ????
    // cout << "upperbound = " << upperbound << endl;

    tb.Start();

    for (size_t i = 0; i < n; i += SIMD<double>::Size())
      {
        SIMD<double> l = 0.0, u = upperbound;
        SIMD<int64_t> evnr = SIMD<int64_t>(n-i-1)-SIMD<int64_t>::FirstInt(); // start with the largest
        for (int bisectit = 0; bisectit < 50; bisectit++)  // 1/2^50 < floating pnt prec
          {
            SIMD<double> mid = 0.5 * (u+l);
            SIMD<int64_t> negcnt = NegCount (B, mid*mid);
            u = If(negcnt > evnr, mid, u);
            l = If(negcnt > evnr, l, mid);
          }
        SIMD<mask64> mask(n-i);
        l.Store (&singvals[i], mask);
      }

    tb.Stop();
    tb.AddFlops (n * 50 * n);
    */

    CalcSValsBiDiagonal(B, singvals);

    // B^T B = L D L^T
    Vector D(n), L(n-1);
    for (int i = 0; i < n; i++)
      D(i) = sqr(B(i,i));
    for (int i = 0; i < n-1; i++)
      L(i) = B(i,i+1) / B(i,i);

    // B = U Sigma V
    // Matrix U(m,m);
    // Matrix V(n,n);
    U = 0.0;

    Vector gamma(n);
    Vector Lp(n-1), Dp(n);
    Vector Um(n-1), Dm(n);
    Vector z(n);   // the eigenvector

    for (int lamnr = 0; lamnr < n; lamnr++)
      {
        // cout << "compute sing " << lamnr << ": " << singvals[lamnr]
        // << " ^ 2 = " << sqr(singvals[lamnr]) << endl;
        tbi.Start();
        double lam = sqr(singvals[lamnr]);

        getvec(L, D, lam, z, 1);

        /*
        dstqds (D, L, lam, Dp, Lp, gamma);
        // cout << "Dp = " << Dp << endl;
        // cout << "Lp = " << Lp << endl;

        dqds (D, L, lam, Dm, Um, gamma);
        // cout << "gamma = " << gamma << endl;

        int r = 0;
        for (int i = 0; i < n; i++)
          if (fabs(gamma(i)) < gamma(r))
            r = i;

        z(r) = 1;
        for (int i = r-1; i >= 0; i--)
          {
            if (Dp(i) != 0)
              z(i) = -Lp(i)*z(i+1);
            else
              z(i) = -(D(i+1)*L(i+1)/(D(i)*L(i))) * z(i+2);
          }
        for (int i = r; i < n-1; i++)
          {
            if (Dm(i+1) != 0)
              z(i+1) = -Um(i)*z(i);
            else
              z(i+1) = -(D(i-1)*L(i-1)/(D(i)*L(i))) * z(i-1);
          }
        // cout << "z = " << z << endl;
        z /= L2Norm(z);
        */

        /*
        // simple version for gamma:
        Vector gamma1(n);
        for (int i = 0; i < n-1; i++)
        gamma1(i) = Dp(i) - sqr(D(i)*L(i)) / Dm(i+1);
        gamma1(n-1) = Dp(n-1);
        cout << "gamma =?= " << gamma1 << endl;
        */


        tbi.Stop();
        tub.Start();
        V.Col(lamnr) = z;
        // U.Col(lamnr) = B * z;
        U.Col(lamnr) = 0.0;
        for (int i = 0; i < n; i++)
          U.Col(lamnr)(i) = B(i,i) * z(i);
        for (int i = 0; i < n-1; i++)
          U.Col(lamnr)(i) += B(i,i+1) * z(i+1);
        U.Col(lamnr) /= singvals[lamnr];
        tub.Stop();
        // cout << "|Ui| = " << L2Norm(U.Col(lamnr)) << endl;
      }

    // cout << "V^T * V = " << Trans(V)*V << endl;
    // cout << "U^T * U = " << Trans(U)*U << endl;
  }



  // T_{GK} - mu = LDL^T
  // Grosser, Lang - On Symmetric Eigenproblems induced by the bidiagonal svd
  // used in MR3 algorithm
  void LDLFromGK(FlatVector<> GK, FlatVector<> L, FlatVector<> D, double mu)
  {
    static Timer tinit("MR3_initldl");
    RegionTimer reg(tinit);

    size_t n = D.Size();
    D(0) = -mu;
    for (int j = 0; j < n-1; j++) {
      L(j) = GK(j) / D(j);
      D(j+1) = - GK(j) * L(j) - mu;
    }
  }



  // THOMPSN
  // needed for MR3
  void CalcSVecsMR3GK(FlatVector<> evals,
                    SliceMatrix<double,ColMajor> U,
                    SliceMatrix<double,ColMajor> V,
                    FlatVector<> L, FlatVector<> D,
                    int cluster_offset)
  {
//    static Timer tgetvec("getvec in recursive CalcSVecsMR3GK");
//    static Timer trefine("refine evals");
    static Timer tldl("MR3_fac");

    size_t n = D.Size();
    size_t k = evals.Size();

    // mark evals whenever the next singval is in the same cluster
    BitArray clustered(k);
    double gaptol = 1e-3;
    for (int i = 0; i < k - 1; i++) {
      if (reldist2(evals[i], evals[i+1]) < gaptol) // cluster size from willems, lang, voemel 2006
        clustered.SetBit(i);
      else
        clustered.Clear(i);
    }
    clustered.Clear(k-1); // simplifies for-loop


    int cluster0 = -1; // -1 as long as we are not in a cluster
    bool in_cluster = 0;
    for (int lamnr = 0; lamnr < k; lamnr++) {
      if (clustered.Test(lamnr)) {
        if (!in_cluster) {
          cluster0 = lamnr;
          in_cluster = 1;
        }
        continue;
      }
      else {
        if (in_cluster) {
          Vector Lchild = L;
          Vector Dchild = D;

          Vector Lchild2 = L;
          Vector Dchild2 = D;

          tldl.Start();
          dstqds2 (Dchild, Lchild, evals[cluster0]);
          dstqds2 (Dchild2, Lchild2, evals[lamnr]);
          tldl.Stop();

          double mu;
          if (L2Norm(Dchild) < L2Norm(Dchild2)) {
            mu = evals[cluster0];
            for (int i = cluster0; i <= lamnr; i++)
              evals[i] -= mu;

            CalcSVecsMR3GK(evals.Range(cluster0, lamnr+1),
                            U.Cols(cluster0, lamnr+1), V.Cols(cluster0, lamnr+1),
                            Lchild, Dchild, cluster_offset + cluster0);

          }
          else {
            mu = evals[lamnr];
            for (int i = cluster0; i <= lamnr; i++)
              evals[i] -= mu;

            CalcSVecsMR3GK(evals.Range(cluster0, lamnr+1),
                            U.Cols(cluster0, lamnr+1), V.Cols(cluster0, lamnr+1),
                            Lchild2, Dchild2, cluster_offset + cluster0);
          }

          /*
          // Calc new RRR
          double mu = evals[cluster0];

          tldl.Start();
          dstqds (Dchild, Lchild, mu);
          if (isnan(L2Norm(Lchild)))
            cerr << "CalcSVecsMR3 child Warning in " << cluster_offset << " " << cluster0 << "-" << lamnr << endl;
          tldl.Stop();

          // shift and refine eigenvalues
          for (int i = cluster0; i <= lamnr; i++)
            evals[i] -= mu;

          CalcSVecsMR3GK(evals.Range(cluster0, lamnr+1),
                          U.Cols(cluster0, lamnr+1), V.Cols(cluster0, lamnr+1),
                          Lchild, Dchild, cluster_offset + cluster0);
          */

          in_cluster = 0;

          continue;
        }
      }

//      tgetvec.Start();
      getvecGK(L, D, evals[lamnr],
                U.Col(lamnr), V.Col(lamnr),
                cluster_offset+lamnr, 1);
//      tgetvec.Stop();
    }
  }


  // calculate k biggest eigenvalues
  void CalcEvalsGK (FlatVector<> GK, FlatVector<> evals, int k=-1)
  {
    if (k < 0) {
      k = evals.Size();
    }

    int n = (GK.Size() + 1) / 2;

    double upperbound = 0;

    // int m = B.Height();
    for (int i = 0; i < GK.Size(); i++) {
      upperbound = max2(upperbound, fabs(GK(i)));
    }
    upperbound *= 3; // ????

    for (size_t i = 0; i < k; i += SIMD<double>::Size())
    {
      SIMD<double> l = 0.0, u = upperbound;
      SIMD<int64_t> evnr = SIMD<int64_t>(n-i-1)-SIMD<int64_t>::FirstInt(); // start with the largest

      for (int bisectit = 0; bisectit < 50; bisectit++)  // 1/2^53 < floating pnt prec
      {
        SIMD<double> mid = 0.5 * (u+l);
        SIMD<int64_t> negcnt = NegCountGK (GK, mid*mid);
        u = If(negcnt > evnr, mid, u);
        l = If(negcnt > evnr, l, mid);
      }
      SIMD<mask64> mask(k-i);
      l.Store (&evals[i], mask);
    }
  }

  // return number of calculated singular values
  int MR3_GK (FlatVector<> GK, SliceMatrix<double,ColMajor> U,
              SliceMatrix<double,ColMajor> V, FlatVector<> evals,
              double eval_limit=-1)
  {
    int n = GK.Size() + 1;

    Vector z(n);
    Vector D(n), L(n-1);
    Vector Dtmp(n), Ltmp(n-1);

    size_t k = CalcEvalsGKBinary(GK, evals, eval_limit);

    if (k == 0)
      return 0;

    // find clusters
    double gaptol = 1e-3; // cluster size from willems, lang, voemel 2006
    BitArray clustered(k);
    for (int i = 0; i < k - 1; i++) {
      if (reldist2(evals[i], evals[i+1]) < gaptol)
        clustered.SetBit(i);
      else
        clustered.Clear(i);
    }
    clustered.Clear(k-1);

    // THOMPSN
    // f.e. Dhillion, Parlett - Multiple representations to compute orthogonal eigenvectors of symmetric tridiagonal matrices
    bool in_cluster = 0;
    int cluster0 = -1;
    for (int lamnr = 0; lamnr < k; lamnr++) {

      if (clustered.Test(lamnr)) {
        if (!in_cluster) {
          cluster0 = lamnr;
          in_cluster = 1;
        }
        continue;
      }

      if (in_cluster) {

        LDLFromGK(GK, L, D, evals[lamnr]);
        LDLFromGK(GK, Ltmp, Dtmp, evals[cluster0]);

        Vector evals_tmp = evals.Range(cluster0, lamnr+1);
        if (L2Norm(D) < L2Norm(Dtmp)) {
          double mu = evals[lamnr];
          for (int j=0; j < evals_tmp.Size(); j++)
            evals_tmp[j] -= mu;
          CalcSVecsMR3GK(evals_tmp,
                          U.Cols(cluster0, lamnr+1), V.Cols(cluster0, lamnr+1),
                          L, D, cluster0);
        }
        else {
          double mu = evals[cluster0];
          for (int j=0; j < evals_tmp.Size(); j++)
            evals_tmp[j] -= mu;
          CalcSVecsMR3GK(evals_tmp,
                          U.Cols(cluster0, lamnr+1), V.Cols(cluster0, lamnr+1),
                          Ltmp, Dtmp, cluster0);
        }

        /*
        double mu = evals[cluster0]; // * (1 + 1e-10);
        LDLFromGK(GK, L, D, mu);

        Vector evals_tmp = evals.Range(cluster0, lamnr+1);
        for (int j=0; j < evals_tmp.Size(); j++)
          evals_tmp[j] -= mu;

        CalcSVecsMR3GK(evals_tmp,
                        U.Cols(cluster0, lamnr+1), V.Cols(cluster0, lamnr+1),
                        L, D, cluster0);
        */

        in_cluster = 0;

      }
      else {
        LDLFromGK(GK, L, D, evals[lamnr]);

        getvecGK(L, D, 0, U.Col(lamnr), V.Col(lamnr), 0, 1);

      }
    }

    return k;

  }

  void rot(double f, double g, double& cs, double& sn, double& r, double eps=1e-20)
  {
    if (fabs(f) < eps)
    {
      cs = 0;
      sn = 1;
      r = g;
    }
    else if (fabs(f) > fabs(g))
    {
      double t = g / f;
      double tt = sqrt(1 + sqr(t));
      cs = 1 / tt;
      sn = t * cs;
      r = f * tt;
    }
    else
    {
      double t = f / g;
      double tt = sqrt(1 + sqr(t));
      sn = 1 / tt;
      cs = t * sn;
      r = g * tt;
    }
  }

  // Implicit Zero-Shift QR Algorithm
  // Demmel, Kahan - Accurate Singular Values of Bidiagonal Matrices p.14
  // can be used to get rid of zero (or small) entries in
  template <ORDERING ORD>
  void ZeroShiftQR(SliceMatrix<double, ORD> B,
                    SliceMatrix<double,ColMajor> U,
                    SliceMatrix<double,ColMajor> V,
                    int start=0)
  {
    size_t n = B.Width();

    auto diag = B.Diag(0);
    auto super_diag = B.Diag(1);

    double normmat = L2Norm(B);
    double eps = normmat * 1e-20;

    double cs, sn, r, h, oldsn;
    double oldcs = 1;
    double f = diag(start);
    double g = super_diag(start);

    Matrix Bold(n);
    for (size_t i=start; i < n-1; i++) {
      rot(f, g, cs, sn, r, eps);

      GivensRotation G1(cs, sn, i, i+1);
      G1.Apply(Trans(V), 1);

      if (i != start)
        super_diag(i-1) = oldsn * r;

      f = oldcs * r;
      g = diag(i+1) * sn;
      h = diag(i+1) * cs;

      rot(f, g, cs, sn, r, eps);

      GivensRotation G2(cs, sn, i, i+1);
      G2.Apply(Trans(U), 1);

      diag(i) = r;
      f = h;
      g = super_diag(i+1);

      oldcs = cs;
      oldsn = sn;
    }
    super_diag(n-2) = h * sn;
    diag(n-1) = h * cs;
  }

  int splitMR3(FlatVector<> GK, SliceMatrix<double, ColMajor> U, SliceMatrix<double, ColMajor> V,
               FlatVector<> singvals, double sval_limit, int n1, int& splits)
  {
    int n = U.Height();
    double normmat = L2Norm(GK);
    double eps = normmat * n * 1e-16;

    // split matrix at 0 values in super diagonal
    int i0 = 0;
    int eval_ind = 0;

    int i = 0;
    if (fabs(GK(2*i+1)) < eps)
    {
      if (fabs(GK(2*i)) > sval_limit)
      {
        singvals(eval_ind) = fabs(GK(2*i));
        U(eval_ind, i0) = (GK(2*i) < 0) ? -1 : 1;
        V(eval_ind, i0) = 1;
        eval_ind++;
      }
      i0++;
    }

    for (i=1; i < n1 - 1; i++)
    {
      if (fabs(GK(2*i+1)) < eps) 
      {
        splits++;
        int new_evals = MR3_GK(GK.Range(2*i0, 2*i+1), 
                               U.Cols(eval_ind, n1).Rows(i0, i+1), 
                               V.Cols(eval_ind, n1).Rows(i0, i+1), 
                               singvals.Range(eval_ind, n1), sval_limit);
        i0 = i+1;

        eval_ind += new_evals;

        if (i < n1-2 && fabs(GK(2*i+3)) < eps)
        {
          if (fabs(GK(2*i+2)) > sval_limit)
          {
            singvals(eval_ind) = fabs(GK(2*i+2));
            U(i0, eval_ind) = (GK(2*i+2) < 0) ? -1 : 1;
            V(i0, eval_ind) = 1;
            eval_ind++;
          }
          i0++;
          i++;
        }
      }
    }

    // in case of a zero in the diagonal, the remaining evals are zero due to zeroshiftqr
    eval_ind += MR3_GK(GK.Range(2*i0, 2*n1-1), U.Cols(eval_ind, n1).Rows(i0, n1), V.Cols(eval_ind, n1).Rows(i0, n1), singvals.Range(eval_ind, n1), sval_limit);

    int k;
    if (sval_limit < 0) {
      k = n;
      for (int i=eval_ind; i < n; i++) {
        U(i,i) = 1;
        V(i,i) = 1;
      }
    }
    else {
      k = eval_ind;
    }

    return k;
  }

  void sortSVD(FlatVector<> singvals,
               SliceMatrix<double,ColMajor> U, SliceMatrix<double,ColMajor> V,
               int k)
  {
      int n = U.Height();

      Array<int> index(k);
      for(int i=0; i<k; i++)
        index[i] = i;

      QuickSortI (FlatArray(k, singvals.Data()), index, greater<double>());

      BitArray check(k);
      check.Clear();

      for (int i=0; i<k; i++) {
        if (check.Test(i)) continue;

        int current = i;
        while (1) {
          int next = index[current];

          if (next == i) break;

          swap(singvals[current], singvals[next]);
          for (int j=0; j<n; j++) {
            swap(U(j, current), U(j, next));
            swap(V(j, current), V(j, next));
          }

          check.SetBit(next);

          current = next;
        }
      }    
  }


  // THOMPSN
  // After Williams' PHD ALGORITHM 3.3
  // in this version, we calculate the singular values after splitting
  // only calculate singular values smaller than sval_limit
  template <ORDERING ORD>
  void CalcSVDBiDiagonalMR3GK (SliceMatrix<double, ORD> B,
                          SliceMatrix<double,ColMajor> U,
                          SliceMatrix<double,ColMajor> V,
                          FlatVector<> singvals, double sval_limit=-1)
  {
    static Timer tzero("MR3 Zero shift QR");
    static Timer tgetgk("MR3_getgk");
    static Timer tsetb("MR3_filldiag");

    size_t n = B.Height();

    // Preprocessing
    // willems-lang-2012 p.11 suggests scaling without specifying the 'proper range'

    // avoid zero and small values according to willems-lang-gk-2012 p.11:
    // values > n \eps \|B\|

    double normmat = L2Norm(B);
    double eps = normmat * n * 1e-16;

    // check for zeros in diagonal
    int i = 0;
    bool notzero = 0;
    for (; i < n; i++) {
      if (fabs(B(i,i)) < normmat * n * 1e-16) {
        break;
      }
      notzero = 1;
    }

    int splits = 0;
    int k;

    if (i < n) 
    {
      // count zeros in diagonal
      size_t count = 0;

      Matrix<double,ColMajor> Uzero = Identity(n);
      Matrix<double,ColMajor> Vzero = Identity(n);

      Matrix<double,ColMajor> Uszero(n);
      Matrix<double,ColMajor> Vszero(n);
      Uszero = 0; 
      Vszero = 0;

      // deflate small values in diagonal
      // use zero shift QR to handle zeros in diag
      tzero.Start();
      for (; i < n - count; i++) {
        if (fabs(B(i,i)) < eps) {

          if (i == 0) 
            if (fabs(B(0,1)) < eps)
              continue;
          else if (i == n-1) 
            if (fabs(B(i-1,i) < eps))
              continue;
          else if (fabs(B(i,i+1)) < eps && fabs(B(i+1,i+1)) > eps && fabs(B(i-1,i) < eps))
            continue;

          if (notzero)
            ZeroShiftQR(B, SliceMatrix(Uzero), SliceMatrix(Vzero), 0);
          else
            ZeroShiftQR(B, SliceMatrix(Uzero), SliceMatrix(Vzero), i);

          i--;

          singvals[n-1-count] = 0;
          count++;
        }
        else {
          notzero = 1;
        }
      }

      // Golub-Kahan Matrix
      Vector GK(2*n-1);
      for (i = 0; i < n-1; i++) {
        GK(2*i) = B(i,i);
        GK(2*i+1) = B(i,i+1);
      }
      GK(2*n-2) = B(n-1,n-1);

      k = splitMR3(GK, Uszero, Vszero, singvals, sval_limit, n - count, splits);

      U = Uzero * Uszero.Cols(k);
      V = Vzero * Vszero.Cols(k);
    }
    else
    {
      U = 0;
      V = 0;

      // Golub-Kahan Matrix
      Vector GK(2*n-1);
      for (i = 0; i < n-1; i++) {
        GK(2*i) = B(i,i);
        GK(2*i+1) = B(i,i+1);
      }
      GK(2*n-2) = B(n-1,n-1);

      k = splitMR3(GK, U, V, singvals, sval_limit, n, splits);
    }

    // sort singular vectors and singular values
    if (splits > 0)
      sortSVD(singvals, U, V, k);

    tsetb.Start();
    B = 0;
    for (int i = 0; i < k; i++)
      B(i,i) = singvals[i];
    tsetb.Stop();

  }

  /*
  // THOMPSN
  // needed for MR3
  void CalcSVecsMR3(FlatVector<> evals,
                    SliceMatrix<double,ColMajor> V,
                    FlatVector<> D, FlatVector<> L,
                    int cluster_offset, bool rqi=0)
  {
    static Timer tshift("get shift in cluster mr3-gk");
    static Timer tgetvec("get vector in cluster mr3-gk");

    size_t n = V.Height();
    size_t k = evals.Size();

    // mark evals whenever the next singval is in the same cluster
    BitArray clustered(k);
    for (int i = 0; i < k - 1; i++) {
      if (reldist2(evals[i], evals[i+1]) < 1e-3) // cluster size from willems, lang, voemel 2006
        clustered.SetBit(i);
      else
        clustered.Clear(i);
    }
    clustered.Clear(k-1); // simplifies for-loop

    int cluster0 = -1; // -1 as long as we are not in a cluster
    for (int lamnr = 0; lamnr < k; lamnr++) {

      if (clustered.Test(lamnr)) {
        if (cluster0 == -1)
          cluster0 = lamnr;
        continue;
      }
      else {
        if (cluster0 != -1) {

          Vector Lchild = L;
          Vector Dchild = D;

          // Calculate RRR
          double mu = evals[lamnr];
          dstqds (Dchild, Lchild, mu);

          for (int i = 0; i < k; i++)
            evals[i] -= mu;

          CalcSVecsMR3(evals.Range(cluster0, lamnr+1),
                  V.Cols(cluster0, lamnr+1),
                  Dchild, Lchild, cluster0, 1);

          cluster0 = -1;

          continue;
        }
      }

        // note: arriving here means we found an isolated sval
        tgetvec.Start();
        double lam = evals[lamnr];

        if (rqi)
          RQI(L, D, lam, V.Col(lamnr));
        else
          getvec(L, D, lam, V.Col(lamnr));
        tgetvec.Stop();
    }
  }

  // THOMPSN
  // MR3 algorithm for SVD of bidiagonal
  // Dhillion - A new O(n^2) Algorithm for the Symmetric Tridiagonal Eigenvalue/Eigenvector Problem
  // old version: does not work for zeros in bidiagonal matrix
  template <ORDERING ORD>
  void CalcSVDBiDiagonalMR3 (SliceMatrix<double, ORD> B,
                            SliceMatrix<double,ColMajor> U,
                            SliceMatrix<double,ColMajor> V,
                            FlatVector<> singvals)
  {
    size_t n = B.Width();
    size_t k = singvals.Size();

    // B^T B = L D L^T
    Vector D(n), L(n-1);
    for (int i = 0; i < n; i++)
      D(i) = sqr(B(i,i));
    for (int i = 0; i < n-1; i++)
      L(i) = B(i,i+1) / B(i,i);

    // B = U Sigma V
    // Matrix U(m,m);
    // Matrix V(n,n);
    U = 0.0;

    Vector gamma(n);
    Vector z(n);   // the eigenvector

    Vector cluster_evals(k);
    for (int i = 0; i < k; i++)
      cluster_evals[i] = sqr(singvals[i]); // needed to store evals calculated in cluster-case

    CalcSVecsMR3(cluster_evals, V, D, L, 0);

    for (int lamnr = 0; lamnr < singvals.Size(); lamnr++) {
      U.Col(lamnr) = 0.0;
      for (int i = 0; i < n; i++)
        U.Col(lamnr)(i) = B(i,i) * V(i, lamnr);
      for (int i = 0; i < n-1; i++)
        U.Col(lamnr)(i) += B(i,i+1) * V(i+1, lamnr);
      U.Col(lamnr) /= singvals[lamnr];
    }

    // cout << "V^T * V = " << Trans(V)*V << endl;
    // cout << "U^T * U = " << Trans(U)*U << endl;
  }
  */



  /**************** Low-Rank-Approximation ****************/

  template <ORDERING ORD>
  void LowRankSVDBiDiagonal1 (SliceMatrix<double, ORD> B,
                            FlatMatrix<double,ColMajor> U, FlatMatrix<double,ColMajor> V,
                            FlatVector<> singvals, double sval_limit)
  {
    size_t n = B.Width();

    Vector g (2*n-1);
    for (int i=0; i<n-1; i++)
    {
      g(2*i) = B(i,i);
      g(2*i+1) = B(i,i+1);
    }
    g(2*n-2) = B(n-1,n-1);

    size_t k = CalcEvalsGKBinary(g, singvals, sval_limit);

    for (int lamnr = 0; lamnr < k; lamnr++)
    {
      double lam = singvals[lamnr];

      Vector L(2*n-1);
      Vector D(2*n);
      LDLFromGK(g, L, D, lam);
      getvecGK(L, D, 0, U.Col(lamnr), V.Col(lamnr), 0, 1);

    }

    B = 0;
    for (int i = 0; i < k; i++) {
      B(i,i) = singvals[i];
    }

    // cout << "V^T * V = " << Trans(V)*V << endl;
    // cout << "U^T * U = " << Trans(U)*U << endl;
  }

  // A = U * Sigma * V^T
  // THOMPSN:
  //    calculate sufficient number of singular vectors, such that the maximal
  //    singular value is below eps
  //    if kmax is not sufficiently small, no singular vectors will be calculated
  template <ORDERING ORD>
  int TLowRankSVD (SliceMatrix<double,ORD> A,
                  SliceMatrix<double, ColMajor> U,
                  SliceMatrix<double, ColMajor> V,
                  size_t kmax, double eps, string method)
  {

    Matrix Aorig = A;

    if (A.Width() > A.Height())
      {
        return TLowRankSVD (Trans(A), V, U, kmax, eps, method);
      }

    // cout << "CalcSVD, h = " << A.Height() << ", w = " << A.Width() << endl;
    if (double norm = L2Norm(A); isnan(norm) || isinf(norm))
      {
        cout << "input matrix norm = " << norm << endl;
        cout << "mat = " << A << endl;
        throw Exception ("called SVD with nan-matrix");
      }

    static Timer t("LowRankCalcSVD"); RegionTimer reg(t);
    static Timer tbidiag("BIDIAG");
    static Timer treduce("BIDIAG_reduce");
    static Timer tapply("BIDIAG_apply");

    size_t m = A.Height();
    size_t n = A.Width();

    Matrix<double, ColMajor> U1(m);
    Matrix<double, ColMajor> V1(n);

    // needed for calculating singular values
    Vector singvals(n);
    double sval_limit;

    int num_svec;
    if (min(n,m) < 500) {
      tbidiag.Start();
      treduce.Start();
      ReduceBiDiagonal (A);
      treduce.Stop();

      if (eps != 0) {
        double sigma0 = LargestSval(A);
        sval_limit = eps * sigma0;
        num_svec = n - NegCount (SliceMatrix(A), sqr(sval_limit));
        if (num_svec == 0 || num_svec > kmax)
          return 0;
      }
      else {
        sval_limit = -1;
        num_svec = n;
      }
      //CalcSValsBiDiagonal(A, singvals);

      tapply.Start();
      BidiagTransformationsSmall(A, U1, V1);
      tapply.Stop();
      tbidiag.Stop();

    }
    else {
      // used to store bulge-chasing transformations and bidiag
      Matrix<double,ColMajor> Aband(min(n,m));
      tbidiag.Start();
      treduce.Start();
      ReduceBiDiagonalBlocked<ORD,ColMajor> (A, Aband);
      treduce.Stop();

      if (eps != 0) {
        double sigma0 = LargestSval(SliceMatrix(Aband));
        sval_limit = eps * sigma0;
        num_svec = n - NegCount (SliceMatrix(Aband), sqr(sval_limit));
        if (num_svec == 0 || num_svec > kmax)
          return 0;
      }
      else {
        sval_limit = -1;
        num_svec = n;
      }
      //CalcSValsBiDiagonal(SliceMatrix(Aband), singvals);

      tapply.Start();
      BidiagTransformationsBlocked<ORD>(A, Aband, U1, V1);
      tapply.Stop();
      tbidiag.Stop();
    }

    //cout << singvals << endl;
    //cout << "needed svals according to mr3: " << num_svec << endl;

    Matrix<double,ColMajor> UB(n), VB(n);

    // cout << "diag A = " << endl << A.Diag(0) << endl;
    // cout << "super-diag A = " << endl << A.Diag(1) << endl;

    if (method == "dnc")
    {
      CalcSVDBiDiagonal (A.Rows(n), UB, VB);
    }
    else if (method == "direct")
    {
      LowRankSVDBiDiagonal1(A.Rows(n), UB, VB, singvals, sval_limit);
    }
    else if (method == "mr3")
    {
      static Timer tmr3("MR3");
      tmr3.Start();
      CalcSVDBiDiagonalMR3GK(A.Rows(n), UB, VB, singvals, sval_limit);
      tmr3.Stop();
    }
    else
    {
      static Timer tmr3("MR3");
      tmr3.Start();
      CalcSVDBiDiagonalMR3GK(A.Rows(n), UB, VB, singvals, sval_limit);
      tmr3.Stop();
    }

    {
      static Timer tmultUV("CalcSVD, mult U1*UB, V1*VB");
      RegionTimer regUV(tmultUV);
      tmultUV.AddFlops (n*n*n+m*m*m);
      U = U1.Cols(n)*UB; // TODO: wrong size
      V = V1*VB;
    }

    return num_svec;
  }


  int LowRankSVD (SliceMatrix<double,RowMajor> A,
                SliceMatrix<double, ColMajor> U,
                SliceMatrix<double, ColMajor> V,
                size_t kmax, double eps, string method)
  {
    return TLowRankSVD (A, U, V, kmax, eps, method);
  }
  int LowRankSVD (SliceMatrix<double,ColMajor> A,
                SliceMatrix<double, ColMajor> U,
                SliceMatrix<double, ColMajor> V,
                size_t kmax, double eps, string method)
  {
    return TLowRankSVD (A, U, V, kmax, eps, method);
  }


  // TODO: remove
  // just for testing
  template <ORDERING ORD>
  void TBidiag (SliceMatrix<double,ORD> A,
                 SliceMatrix<double, ColMajor> U,
                 SliceMatrix<double, ColMajor> V,
                 bool blocked=0)
  {

    Matrix A_orig = A;

    if (A.Width() > A.Height())
      {
        TBidiag (Trans(A), V, U, blocked);
      }

    // cout << "CalcSVD, h = " << A.Height() << ", w = " << A.Width() << endl;
    if (double norm = L2Norm(A); isnan(norm) || isinf(norm))
      {
        cout << "input matrix norm = " << norm << endl;
        cout << "mat = " << A << endl;
        throw Exception ("called SVD with nan-matrix");
      }

    static Timer t("LowRankCalcSVD"); RegionTimer reg(t);
    size_t m = A.Height();
    size_t n = A.Width();

    Matrix<double, ColMajor> U1(m);
    Matrix<double, ColMajor> V1(n);

    // needed for calculating singular values
    Vector singvals(n);

    int num_svec;
    if (blocked) {
      // used to store bulge-chasing transformations and bidiag
      Matrix<double,ColMajor> Aband(min(n,m));
      ReduceBiDiagonalBlocked<ORD,ColMajor> (A, Aband);
      BidiagTransformationsBlocked<ORD>(A, Aband, U, V);
    }
    else {
      ReduceBiDiagonal (A);
      BidiagTransformationsSmall(A, U, V);
    }

  }

  void Bidiagonalize (SliceMatrix<double,RowMajor> A,
                SliceMatrix<double, ColMajor> U,
                SliceMatrix<double, ColMajor> V,
                bool blocked=0)
  {
    return TBidiag(A, U, V, blocked);
  }

  void Bidiagonalize (SliceMatrix<double,ColMajor> A,
                SliceMatrix<double,ColMajor> U,
                SliceMatrix<double,ColMajor> V,
                bool blocked=0)
  {
    return TBidiag(A, U, V, blocked);
  }


  // TODO: remove
  template <ORDERING ORD>
  void TDiagonalize (SliceMatrix<double, ORD> A,
                  SliceMatrix<double, ColMajor> U,
                  SliceMatrix<double, ColMajor> V)
  {
    Matrix A_orig = A;

    if (A.Width() > A.Height())
      {
        return Diagonalize (Trans(A), V, U);
      }

    // cout << "CalcSVD, h = " << A.Height() << ", w = " << A.Width() << endl;
    if (double norm = L2Norm(A); isnan(norm) || isinf(norm))
      {
        cout << "input matrix norm = " << norm << endl;
        cout << "mat = " << A << endl;
        throw Exception ("called SVD with nan-matrix");
      }

    size_t m = A.Height();
    size_t n = A.Width();

    Vector singvals(n);

    CalcSVDBiDiagonalMR3GK(A.Rows(n), U, V, singvals);

    A.Rows(n,m) = 0;

  }

  void Diagonalize (SliceMatrix<double, RowMajor> A,
                  SliceMatrix<double, ColMajor> U,
                  SliceMatrix<double, ColMajor> V)
  {
    TDiagonalize(A, U, V);
  }

  void Diagonalize (SliceMatrix<double, ColMajor> A,
                  SliceMatrix<double, ColMajor> U,
                  SliceMatrix<double, ColMajor> V)
  {
    TDiagonalize(A, U, V);
  }



}
