/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_CARTESIANGRID_HPP
#define CABANA_CARTESIANGRID_HPP

#include <Kokkos_Core.hpp>

#include <limits>
#include <type_traits>

#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))

namespace Cabana
{
namespace Impl
{
//! \cond Impl

template <class Real, int NumDim = 3, typename std::enable_if<
                          std::is_floating_point<Real>::value, int>::type = 0>
class CartesianGrid
{
  public:
    using real_type = Real;

    Real _min[NumDim];
    Real _max[NumDim];
    Real _delta[NumDim];
    Real _rdx[NumDim];
    int _n[NumDim];

    CartesianGrid() {}

    CartesianGrid( const Real min[NumDim], const Real max[NumDim],
                   const Real delta[NumDim])
    {
        for ( int dim = 0; dim < NumDim; ++dim )
        {
            _min[dim] = min[dim];
            _max[dim] = max[dim];
            _n[dim] = cellsBetween( max[dim], min[dim], 1.0/delta[dim] );
            _delta[dim] = (_max[dim] - min[dim]) / _n[dim];
            _rdx[dim] = 1.0 / delta[dim];
        }
    }

    // Get the total number of cells.
    KOKKOS_INLINE_FUNCTION
    std::size_t totalNumCells() const {
        std::size_t total = 1;
        for ( int dim = 0; dim < NumDim; ++dim )
            total *= _n[dim];
        return total;
        }

    // Get the number of cells in each direction.
    KOKKOS_INLINE_FUNCTION
    void numCells( int (&num)[NumDim] )
    {
        for ( int dim = 0; dim < NumDim; ++dim )
            num[dim] = _n[dim];
    }

    // Get the number of cells in a given direction.
    KOKKOS_INLINE_FUNCTION
    int numBin( const int dim ) const
    {
        if ( dim < 0 || dim >= NumDim )
            return -1;
        else
            return _n[dim];
    }

    // Given a position get the ijk indices of the cell in which
    KOKKOS_INLINE_FUNCTION
    void locatePoint( const Real p[NumDim], int (&c)[NumDim] ) const
    {
        // Since we use a floor function a point on the outer boundary
        // will be found in the next cell, causing an out of bounds error
        for ( int dim = 0; dim < NumDim; ++dim )
        {
            c[dim] = cellsBetween( p[dim], _min[dim], _rdx[dim] );
            c[dim] = ( c[dim] == _n[dim] ) ? c[dim] - 1 : c[dim];
        }
    }

    // Given a position and a cell index get square of the minimum distance to
    // that point to any point in the cell. If the point is in the cell the
    // returned distance is zero.
    KOKKOS_INLINE_FUNCTION
    Real minDistanceToPoint( const Real p[NumDim], const int c[NumDim] ) const
    {
        Real dist_squared = 0.0;
        for( int dim = 0; dim < NumDim; ++dim )
        {
            Real x = _min[dim] + ( c[dim] + 0.5 ) * _delta[dim];
            Real rx = fabs( p[dim] - x ) - 0.5 * _delta[dim];
            rx = ( rx > 0.0 ) ? rx : 0.0;
            dist_squared += rx * rx;
        }

        return dist_squared;
    }

    // Given the ijk index of a cell get its cardinal index.
    KOKKOS_INLINE_FUNCTION
    int cardinalCellIndex( const int i[NumDim] ) const
    {
        //For 3D, this is the same as
        //(i * _ny + j) * _nz + k;
        int cardinal = 0;
        for ( int dim = 0; dim < NumDim; ++dim )
            cardinal = cardinal * _n[dim] + i[dim];
        return cardinal;
    }

    KOKKOS_INLINE_FUNCTION
    void ijkBinIndex( const int cardinal, int (&i)[NumDim] ) const
    {
        int k = cardinal;
        for ( int dim = NumDim - 1; dim >= 0; --dim )
        {
            i[dim] = k % _n[dim];
            k /= _n[dim];
        }
    }

    // Calculate the number of full cells between 2 points.
    KOKKOS_INLINE_FUNCTION
    int cellsBetween( const Real max, const Real min, const Real rdelta ) const
    {
        return Kokkos::floor( ( max - min ) * rdelta );
    }

    /// ------------------------------------------------------------------------
    // Implement 3D functions only (for backward compatibility).
    /// ------------------------------------------------------------------------
    CartesianGrid( const Real min_x, const Real min_y, const Real min_z,
                   const Real max_x, const Real max_y, const Real max_z,
                   const Real delta_x, const Real delta_y, const Real delta_z ){
        assertm(NumDim == 3 , "CartesianGrid: deprecated constructor only works for 3D");
        Real min[3] = {min_x, min_y, min_z};
        Real max[3] = {max_x, max_y, max_z};
        Real delta[3] = {delta_x, delta_y, delta_z};
        for ( int dim = 0; dim < NumDim; ++dim )
        {
            _min[dim] = min[dim];
            _max[dim] = max[dim];
            _n[dim] = cellsBetween( max[dim], min[dim], 1.0/delta[dim] );
            _delta[dim] = (_max[dim] - min[dim]) / _n[dim];
            _rdx[dim] = 1.0 / delta[dim];
        }
    }
    KOKKOS_INLINE_FUNCTION
    void numCells( int& num_x, int& num_y, int& num_z ){
        assertm(NumDim == 3 , "CartesianGrid: deprecated function only works for 3D");
        int num[3];
        numCells(num);
        num_x = num[0];
        num_y = num[1];
        num_z = num[2];
    }
    KOKKOS_INLINE_FUNCTION
    void locatePoint( const Real xp, const Real yp, const Real zp, int& ic,
                      int& jc, int& kc ) const
    {
        assertm(NumDim == 3 , "CartesianGrid: deprecated function only works for 3D");
        Real p[3] = {xp, yp, zp};
        int c[3];
        locatePoint(p, c);
        ic = c[0];
        jc = c[1];
        kc = c[2];
    }
    KOKKOS_INLINE_FUNCTION
    Real minDistanceToPoint( const Real xp, const Real yp, const Real zp,
                             const int ic, const int jc, const int kc ) const
    {
        assertm(NumDim == 3 , "CartesianGrid: deprecated function only works for 3D");
        Real p[3] = {xp, yp, zp};
        int c[3] = {ic, jc, kc};
        return minDistanceToPoint(p, c);
    }
    KOKKOS_INLINE_FUNCTION
    int cardinalCellIndex( const int i, const int j, const int k ) const
    {
        assertm(NumDim == 3 , "CartesianGrid: deprecated function only works for 3D");
        int ijk[3] = {i, j, k};
        return cardinalCellIndex(ijk);
    }
    KOKKOS_INLINE_FUNCTION
    void ijkBinIndex( const int cardinal, int& i, int& j, int& k ) const
    {
        assertm(NumDim == 3 , "CartesianGrid: deprecated function only works for 3D");
        int ijk[3];
        ijkBinIndex(cardinal, ijk);
        i = ijk[0];
        j = ijk[1];
        k = ijk[2];
    }
};

//! \endcond
} // end namespace Impl
} // end namespace Cabana

#endif // end CABANA_CARTESIANGRID_HPP
