/*===========================================================================*\
 *                                                                           *
 *                            OpenVolumeMesh                                 *
 *        Copyright (C) 2011 by Computer Graphics Group, RWTH Aachen         *
 *                        www.openvolumemesh.org                             *
 *                                                                           *
 *---------------------------------------------------------------------------*
 *  This file is part of OpenVolumeMesh.                                     *
 *                                                                           *
 *  OpenVolumeMesh is free software: you can redistribute it and/or modify   *
 *  it under the terms of the GNU Lesser General Public License as           *
 *  published by the Free Software Foundation, either version 3 of           *
 *  the License, or (at your option) any later version with the              *
 *  following exceptions:                                                    *
 *                                                                           *
 *  If other files instantiate templates or use macros                       *
 *  or inline functions from this file, or you compile this file and         *
 *  link it with other files to produce an executable, this file does        *
 *  not by itself cause the resulting executable to be covered by the        *
 *  GNU Lesser General Public License. This exception does not however       *
 *  invalidate any other reasons why the executable file might be            *
 *  covered by the GNU Lesser General Public License.                        *
 *                                                                           *
 *  OpenVolumeMesh is distributed in the hope that it will be useful,        *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            *
 *  GNU Lesser General Public License for more details.                      *
 *                                                                           *
 *  You should have received a copy of the GNU LesserGeneral Public          *
 *  License along with OpenVolumeMesh.  If not,                              *
 *  see <http://www.gnu.org/licenses/>.                                      *
 *                                                                           *
\*===========================================================================*/

/*===========================================================================*\
 *                                                                           *
 *   $Revision$                                                         *
 *   $Date$                    *
 *   $LastChangedBy$                                                *
 *                                                                           *
\*===========================================================================*/

#ifndef ITERATORS_HH_
#define ITERATORS_HH_

#include <iterator>
#include <set>
#include <vector>

#include "OpenVolumeMeshHandle.hh"

namespace OpenVolumeMesh {

class TopologyKernel;

template <
class OH /* Output handle type */>
class BaseIterator {
public:

	// STL compliance
	typedef std::bidirectional_iterator_tag iterator_category;
	typedef int						        difference_type;
	typedef const OH				        value_type;
	typedef const OH*				        pointer;
	typedef const OH&				        reference;


    BaseIterator(const TopologyKernel* _mesh, const OH& _ch) :
        valid_(true), cur_handle_(_ch), mesh_(_mesh) {}

    BaseIterator(const TopologyKernel* _mesh) :
        valid_(true), mesh_(_mesh) {}

    // STL compliance (needs to have default constructor)
	BaseIterator() : valid_(false), mesh_(0) {}
	virtual ~BaseIterator() {}
	bool operator== (const BaseIterator& _c) const {
        return (this->cur_handle_ == _c.cur_handle() &&
                this->valid_ == _c.valid() &&
				this->mesh_ == _c.mesh());
	}
	bool operator!= (const BaseIterator& _c) const {
		return !this->operator==(_c);
	}

	pointer operator->() const {
		return &cur_handle_;
	}

	reference operator*() const {
		return cur_handle_;
	}

	bool operator< (const BaseIterator& _c) const {
	    return cur_handle_.idx() < _c.cur_handle_.idx();
	}

	BaseIterator& operator=(const BaseIterator& _c) {
		this->valid_ = _c.valid();
        this->cur_handle_ = _c.cur_handle();
		this->mesh_ = _c.mesh();
		return *this;
	}

	operator bool() const {
		return valid_;
	}

	void valid(bool _valid) {
		valid_ = _valid;
    }
    bool valid() const {
        return valid_;
    }
	void cur_handle(const OH& _h) {
		cur_handle_ = _h;
	}
	reference cur_handle() const {
		return cur_handle_;
    }
	const TopologyKernel* mesh() const {
		return mesh_;
	}

private:

    bool valid_;
    OH cur_handle_;
	const TopologyKernel* mesh_;
};

template <
class IH /*  Input handle type */,
class OH /* Output handle type */>
class BaseCirculator : public BaseIterator<OH> {
public:

    typedef BaseIterator<OH> BaseIter;

    BaseCirculator(const TopologyKernel* _mesh, const IH& _ih, const OH& _oh, int _max_laps = 1) :
        BaseIter(_mesh, _oh),
        lap_(0),
        max_laps_(_max_laps),
        ref_handle_(_ih)
    {}

    BaseCirculator(const TopologyKernel* _mesh, const IH& _ih, int _max_laps = 1) :
        BaseIter(_mesh, OH()),
        lap_(0),
        max_laps_(_max_laps),
        ref_handle_(_ih)
    {}

    // STL compliance (needs to have default constructor)
    BaseCirculator() :
        BaseIter(),
        lap_(0),
        max_laps_(1)
    {}

    virtual ~BaseCirculator() {}

    bool operator== (const BaseCirculator& _c) const {
        return (BaseIter::operator==(_c) &&
                this->lap() == _c.lap() &&
                this->ref_handle() == _c.ref_handle());
    }
    bool operator!= (const BaseCirculator& _c) const {
        return !this->operator==(_c);
    }

    bool operator< (const BaseCirculator& _c) const {
        if (lap_ == _c.lap_)
            return BaseIter::operator<(_c);
        else
            return lap_ < _c.lap_;
    }

    BaseCirculator& operator=(const BaseCirculator& _c) {
        BaseIter::operator=(_c);
        this->ref_handle_ = _c.ref_handle();
        this->lap_ = _c.lap_;
        this->max_laps_ = _c.max_laps_;
        return *this;
    }

    const IH& ref_handle() const {
        return ref_handle_;
    }

    void lap(int _lap) {
        lap_ = _lap;
    }
    int lap() const {
        return lap_;
    }

    void max_laps(int _max_laps) {
        max_laps_ = _max_laps;
    }
    int max_laps() const {
        return max_laps_;
    }

protected:
    int lap_;
    int max_laps_;
    IH ref_handle_;

};

//===========================================================================

class VertexOHalfEdgeIter :
    public BaseCirculator<
	VertexHandle,
	HalfEdgeHandle> {
public:
    typedef BaseCirculator<
			VertexHandle,
			HalfEdgeHandle> BaseIter;


	VertexOHalfEdgeIter(const VertexHandle& _vIdx,
            const TopologyKernel* _mesh, int _max_laps = 1);

	// Post increment/decrement operator
	VertexOHalfEdgeIter operator++(int) {
		VertexOHalfEdgeIter cpy = *this;
		++(*this);
		return cpy;
	}
	VertexOHalfEdgeIter operator--(int) {
		VertexOHalfEdgeIter cpy = *this;
		--(*this);
		return cpy;
	}
	VertexOHalfEdgeIter operator+(int _n) {
		VertexOHalfEdgeIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			++cpy;
		}
		return cpy;
	}
	VertexOHalfEdgeIter operator-(int _n) {
		VertexOHalfEdgeIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			--cpy;
		}
		return cpy;
	}
	VertexOHalfEdgeIter& operator+=(int _n) {
		for(int i = 0; i < _n; ++i) {
			++(*this);
		}
		return *this;
	}
	VertexOHalfEdgeIter& operator-=(int _n) {
		for(int i = 0; i < _n; ++i) {
			--(*this);
		}
		return *this;
	}

	VertexOHalfEdgeIter& operator++();
	VertexOHalfEdgeIter& operator--();

private:

    size_t cur_index_;
};

//===========================================================================

class HalfEdgeHalfFaceIter : public BaseCirculator<
	HalfEdgeHandle,
	HalfFaceHandle> {
public:
    typedef BaseCirculator<
			HalfEdgeHandle,
			HalfFaceHandle> BaseIter;


    HalfEdgeHalfFaceIter(const HalfEdgeHandle& _heIdx, const TopologyKernel* _mesh, int _max_laps);

	// Post increment/decrement operator
	HalfEdgeHalfFaceIter operator++(int) {
		HalfEdgeHalfFaceIter cpy = *this;
		++(*this);
		return cpy;
	}
	HalfEdgeHalfFaceIter operator--(int) {
		HalfEdgeHalfFaceIter cpy = *this;
		--(*this);
		return cpy;
	}
	HalfEdgeHalfFaceIter operator+(int _n) {
		HalfEdgeHalfFaceIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			++cpy;
		}
		return cpy;
	}
	HalfEdgeHalfFaceIter operator-(int _n) {
		HalfEdgeHalfFaceIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			--cpy;
		}
		return cpy;
	}
	HalfEdgeHalfFaceIter& operator+=(int _n) {
		for(int i = 0; i < _n; ++i) {
			++(*this);
		}
		return *this;
	}
	HalfEdgeHalfFaceIter& operator-=(int _n) {
		for(int i = 0; i < _n; ++i) {
			--(*this);
		}
		return *this;
	}

	HalfEdgeHalfFaceIter& operator++();
	HalfEdgeHalfFaceIter& operator--();

private:
    size_t cur_index_;
};

//===========================================================================

class VertexCellIter : public BaseCirculator<
	VertexHandle,
	CellHandle> {
public:
    typedef BaseCirculator<
			VertexHandle,
			CellHandle> BaseIter;

    VertexCellIter(const VertexHandle& _vIdx, const TopologyKernel* _mesh, int _max_laps = 1);

	// Post increment/decrement operator
	VertexCellIter operator++(int) {
		VertexCellIter cpy = *this;
		++(*this);
		return cpy;
	}
	VertexCellIter operator--(int) {
		VertexCellIter cpy = *this;
		--(*this);
		return cpy;
	}
	VertexCellIter operator+(int _n) {
		VertexCellIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			++cpy;
		}
		return cpy;
	}
	VertexCellIter operator-(int _n) {
		VertexCellIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			--cpy;
		}
		return cpy;
	}
	VertexCellIter& operator+=(int _n) {
		for(int i = 0; i < _n; ++i) {
			++(*this);
		}
		return *this;
	}
	VertexCellIter& operator-=(int _n) {
		for(int i = 0; i < _n; ++i) {
			--(*this);
		}
		return *this;
	}

	VertexCellIter& operator++();
	VertexCellIter& operator--();

private:
    std::vector<CellHandle> cells_;
    size_t cur_index_;
};

class HalfEdgeCellIter : public BaseCirculator<
	HalfEdgeHandle,
	CellHandle> {
public:
    typedef BaseCirculator<
			HalfEdgeHandle,
			CellHandle> BaseIter;


    HalfEdgeCellIter(const HalfEdgeHandle& _heIdx, const TopologyKernel* _mesh, int _max_laps = 1);

	// Post increment/decrement operator
	HalfEdgeCellIter operator++(int) {
		HalfEdgeCellIter cpy = *this;
		++(*this);
		return cpy;
	}
	HalfEdgeCellIter operator--(int) {
		HalfEdgeCellIter cpy = *this;
		--(*this);
        return cpy;
    }
	HalfEdgeCellIter operator+(int _n) {
		HalfEdgeCellIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			++cpy;
		}
		return cpy;
	}
	HalfEdgeCellIter operator-(int _n) {
		HalfEdgeCellIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			--cpy;
		}
		return cpy;
	}
	HalfEdgeCellIter& operator+=(int _n) {
		for(int i = 0; i < _n; ++i) {
			++(*this);
		}
		return *this;
	}
	HalfEdgeCellIter& operator-=(int _n) {
		for(int i = 0; i < _n; ++i) {
			--(*this);
		}
		return *this;
	}

	HalfEdgeCellIter& operator++();
	HalfEdgeCellIter& operator--();

private:
    CellHandle getCellHandle(int _cur_index) const;

private:
    std::vector<CellHandle> cells_;
    size_t cur_index_;
};

//===========================================================================

class CellVertexIter : public BaseCirculator<
	CellHandle,
	VertexHandle> {
public:
    typedef BaseCirculator<
			CellHandle,
			VertexHandle> BaseIter;

    CellVertexIter(const CellHandle& _cIdx, const TopologyKernel* _mesh, int _max_laps = 1);

	// Post increment/decrement operator
	CellVertexIter operator++(int) {
		CellVertexIter cpy = *this;
		++(*this);
		return cpy;
	}
	CellVertexIter operator--(int) {
		CellVertexIter cpy = *this;
		--(*this);
		return cpy;
	}
	CellVertexIter operator+(int _n) {
		CellVertexIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			++cpy;
		}
		return cpy;
	}
	CellVertexIter operator-(int _n) {
		CellVertexIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			--cpy;
		}
		return cpy;
	}
	CellVertexIter& operator+=(int _n) {
		for(int i = 0; i < _n; ++i) {
			++(*this);
		}
		return *this;
	}
	CellVertexIter& operator-=(int _n) {
		for(int i = 0; i < _n; ++i) {
			--(*this);
		}
		return *this;
	}

	CellVertexIter& operator++();
	CellVertexIter& operator--();

private:
	std::vector<VertexHandle> incident_vertices_;
    size_t cur_index_;
};

//===========================================================================

class CellCellIter : public BaseCirculator<
	CellHandle,
	CellHandle> {
public:
    typedef BaseCirculator<
			CellHandle,
			CellHandle> BaseIter;

    CellCellIter(const CellHandle& _cIdx, const TopologyKernel* _mesh, int _max_laps = 1);

	// Post increment/decrement operator
	CellCellIter operator++(int) {
		CellCellIter cpy = *this;
		++(*this);
		return cpy;
	}
	CellCellIter operator--(int) {
		CellCellIter cpy = *this;
		--(*this);
		return cpy;
	}
	CellCellIter operator+(int _n) {
		CellCellIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			++cpy;
		}
		return cpy;
	}
	CellCellIter operator-(int _n) {
		CellCellIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			--cpy;
		}
		return cpy;
	}
	CellCellIter& operator+=(int _n) {
		for(int i = 0; i < _n; ++i) {
			++(*this);
		}
		return *this;
	}
	CellCellIter& operator-=(int _n) {
		for(int i = 0; i < _n; ++i) {
			--(*this);
		}
		return *this;
	}

	CellCellIter& operator++();
	CellCellIter& operator--();

private:
    std::vector<CellHandle> adjacent_cells_;
    size_t cur_index_;
};

//===========================================================================

class HalfFaceVertexIter : public BaseCirculator<
    HalfFaceHandle,
    VertexHandle> {
public:
    typedef BaseCirculator<
            HalfFaceHandle,
            VertexHandle> BaseIter;

    HalfFaceVertexIter(const HalfFaceHandle& _hIdx, const TopologyKernel* _mesh, int _max_laps = 1);

    // Post increment/decrement operator
    HalfFaceVertexIter operator++(int) {
        HalfFaceVertexIter cpy = *this;
        ++(*this);
        return cpy;
    }
    HalfFaceVertexIter operator--(int) {
        HalfFaceVertexIter cpy = *this;
        --(*this);
        return cpy;
    }
    HalfFaceVertexIter operator+(int _n) {
        HalfFaceVertexIter cpy = *this;
        for(int i = 0; i < _n; ++i) {
            ++cpy;
        }
        return cpy;
    }
    HalfFaceVertexIter operator-(int _n) {
        HalfFaceVertexIter cpy = *this;
        for(int i = 0; i < _n; ++i) {
            --cpy;
        }
        return cpy;
    }
    HalfFaceVertexIter& operator+=(int _n) {
        for(int i = 0; i < _n; ++i) {
            ++(*this);
        }
        return *this;
    }
    HalfFaceVertexIter& operator-=(int _n) {
        for(int i = 0; i < _n; ++i) {
            --(*this);
        }
        return *this;
    }

    HalfFaceVertexIter& operator++();
    HalfFaceVertexIter& operator--();

private:
    std::vector<VertexHandle> vertices_;
    size_t cur_index_;
};

//===========================================================================

class BoundaryHalfFaceHalfFaceIter : public BaseCirculator<HalfFaceHandle,
    HalfFaceHandle> {
private:
    typedef BaseCirculator<HalfFaceHandle,
            HalfFaceHandle> BaseIter;
public:
    BoundaryHalfFaceHalfFaceIter(const HalfFaceHandle& _ref_h,
            const TopologyKernel* _mesh, int _max_laps = 1);

    // Post increment/decrement operator
    BoundaryHalfFaceHalfFaceIter operator++(int) {
        BoundaryHalfFaceHalfFaceIter cpy = *this;
        ++(*this);
        return cpy;
    }
    BoundaryHalfFaceHalfFaceIter operator--(int) {
        BoundaryHalfFaceHalfFaceIter cpy = *this;
        --(*this);
        return cpy;
    }
    BoundaryHalfFaceHalfFaceIter operator+(int _n) {
        BoundaryHalfFaceHalfFaceIter cpy = *this;
        for(int i = 0; i < _n; ++i) {
            ++cpy;
        }
        return cpy;
    }
    BoundaryHalfFaceHalfFaceIter operator-(int _n) {
        BoundaryHalfFaceHalfFaceIter cpy = *this;
        for(int i = 0; i < _n; ++i) {
            --cpy;
        }
        return cpy;
    }
    BoundaryHalfFaceHalfFaceIter& operator+=(int _n) {
        for(int i = 0; i < _n; ++i) {
            ++(*this);
        }
        return *this;
    }
    BoundaryHalfFaceHalfFaceIter& operator-=(int _n) {
        for(int i = 0; i < _n; ++i) {
            --(*this);
        }
        return *this;
    }

    const EdgeHandle& common_edge() const { return common_edges_[cur_index_]; }

    BoundaryHalfFaceHalfFaceIter& operator++();
    BoundaryHalfFaceHalfFaceIter& operator--();

private:
    std::vector<HalfFaceHandle> neighbor_halffaces_;
    std::vector<EdgeHandle> common_edges_;
    size_t cur_index_;
};

//===========================================================================

class VertexIter : public BaseIterator<VertexHandle> {
public:
    typedef BaseIterator<VertexHandle> BaseIter;


    VertexIter(const TopologyKernel* _mesh, const VertexHandle& _vh = VertexHandle(0));

	// Post increment/decrement operator
	VertexIter operator++(int) {
		VertexIter cpy = *this;
		++(*this);
		return cpy;
	}
	VertexIter operator--(int) {
		VertexIter cpy = *this;
		--(*this);
		return cpy;
	}
	VertexIter operator+(int _n) {
		VertexIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			++cpy;
		}
		return cpy;
	}
	VertexIter operator-(int _n) {
		VertexIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			--cpy;
		}
		return cpy;
	}
	VertexIter& operator+=(int _n) {
		for(int i = 0; i < _n; ++i) {
			++(*this);
		}
		return *this;
	}
	VertexIter& operator-=(int _n) {
		for(int i = 0; i < _n; ++i) {
			--(*this);
		}
		return *this;
	}

	VertexIter& operator++();
	VertexIter& operator--();

private:
    int cur_index_;
};

//===========================================================================

class EdgeIter : public BaseIterator<EdgeHandle> {
public:
    typedef BaseIterator<EdgeHandle> BaseIter;


	EdgeIter(const TopologyKernel* _mesh, const EdgeHandle& _eh = EdgeHandle(0));

	// Post increment/decrement operator
	EdgeIter operator++(int) {
		EdgeIter cpy = *this;
		++(*this);
		return cpy;
	}
	EdgeIter operator--(int) {
		EdgeIter cpy = *this;
		--(*this);
		return cpy;
	}
	EdgeIter operator+(int _n) {
		EdgeIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			++cpy;
		}
		return cpy;
	}
	EdgeIter operator-(int _n) {
		EdgeIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			--cpy;
		}
		return cpy;
	}
	EdgeIter& operator+=(int _n) {
		for(int i = 0; i < _n; ++i) {
			++(*this);
		}
		return *this;
	}
	EdgeIter& operator-=(int _n) {
		for(int i = 0; i < _n; ++i) {
			--(*this);
		}
		return *this;
	}

	EdgeIter& operator++();
	EdgeIter& operator--();

private:
    int cur_index_;
};

//===========================================================================

class HalfEdgeIter : public BaseIterator<HalfEdgeHandle> {
public:
    typedef BaseIterator<HalfEdgeHandle> BaseIter;


	HalfEdgeIter(const TopologyKernel* _mesh, const HalfEdgeHandle& _heh = HalfEdgeHandle(0));

	// Post increment/decrement operator
	HalfEdgeIter operator++(int) {
		HalfEdgeIter cpy = *this;
		++(*this);
		return cpy;
	}
	HalfEdgeIter operator--(int) {
		HalfEdgeIter cpy = *this;
		--(*this);
		return cpy;
	}
	HalfEdgeIter operator+(int _n) {
		HalfEdgeIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			++cpy;
		}
		return cpy;
	}
	HalfEdgeIter operator-(int _n) {
		HalfEdgeIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			--cpy;
		}
		return cpy;
	}
	HalfEdgeIter& operator+=(int _n) {
		for(int i = 0; i < _n; ++i) {
			++(*this);
		}
		return *this;
	}
	HalfEdgeIter& operator-=(int _n) {
		for(int i = 0; i < _n; ++i) {
			--(*this);
		}
		return *this;
	}

	HalfEdgeIter& operator++();
	HalfEdgeIter& operator--();

private:
    int cur_index_;
};

//===========================================================================

class FaceIter : public BaseIterator<FaceHandle> {
public:
    typedef BaseIterator<FaceHandle> BaseIter;


	FaceIter(const TopologyKernel* _mesh, const FaceHandle& _fh = FaceHandle(0));

	// Post increment/decrement operator
	FaceIter operator++(int) {
		FaceIter cpy = *this;
		++(*this);
		return cpy;
	}
	FaceIter operator--(int) {
		FaceIter cpy = *this;
		--(*this);
		return cpy;
	}
	FaceIter operator+(int _n) {
		FaceIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			++cpy;
		}
		return cpy;
	}
	FaceIter operator-(int _n) {
		FaceIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			--cpy;
		}
		return cpy;
	}
	FaceIter& operator+=(int _n) {
		for(int i = 0; i < _n; ++i) {
			++(*this);
		}
		return *this;
	}
	FaceIter& operator-=(int _n) {
		for(int i = 0; i < _n; ++i) {
			--(*this);
		}
		return *this;
	}

	FaceIter& operator++();
	FaceIter& operator--();

private:
    int cur_index_;
};

//===========================================================================

class HalfFaceIter : public BaseIterator<HalfFaceHandle> {
public:
    typedef BaseIterator<HalfFaceHandle> BaseIter;


	HalfFaceIter(const TopologyKernel* _mesh, const HalfFaceHandle& _hfh = HalfFaceHandle(0));

	// Post increment/decrement operator
	HalfFaceIter operator++(int) {
		HalfFaceIter cpy = *this;
		++(*this);
		return cpy;
	}
	HalfFaceIter operator--(int) {
		HalfFaceIter cpy = *this;
		--(*this);
		return cpy;
	}
	HalfFaceIter operator+(int _n) {
		HalfFaceIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			++cpy;
		}
		return cpy;
	}
	HalfFaceIter operator-(int _n) {
		HalfFaceIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			--cpy;
		}
		return cpy;
	}
	HalfFaceIter& operator+=(int _n) {
		for(int i = 0; i < _n; ++i) {
			++(*this);
		}
		return *this;
	}
	HalfFaceIter& operator-=(int _n) {
		for(int i = 0; i < _n; ++i) {
			--(*this);
		}
		return *this;
	}

	HalfFaceIter& operator++();
	HalfFaceIter& operator--();

private:
    int cur_index_;
};

//===========================================================================

class CellIter : public BaseIterator<CellHandle> {
public:
    typedef BaseIterator<CellHandle> BaseIter;


	CellIter(const TopologyKernel* _mesh, const CellHandle& _ch = CellHandle(0));

	// Post increment/decrement operator
	CellIter operator++(int) {
		CellIter cpy = *this;
		++(*this);
		return cpy;
	}
	CellIter operator--(int) {
		CellIter cpy = *this;
		--(*this);
		return cpy;
	}
	CellIter operator+(int _n) {
		CellIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			++cpy;
		}
		return cpy;
	}
	CellIter operator-(int _n) {
		CellIter cpy = *this;
		for(int i = 0; i < _n; ++i) {
			--cpy;
		}
		return cpy;
	}
	CellIter& operator+=(int _n) {
		for(int i = 0; i < _n; ++i) {
			++(*this);
		}
		return *this;
	}
	CellIter& operator-=(int _n) {
		for(int i = 0; i < _n; ++i) {
			--(*this);
		}
		return *this;
	}

	CellIter& operator++();
	CellIter& operator--();

private:
    int cur_index_;
};

//===========================================================================

class BoundaryFaceIter : public BaseIterator<FaceHandle> {
public:
    typedef BaseIterator<FaceHandle> BaseIter;


    BoundaryFaceIter(const TopologyKernel* _mesh);

    // Post increment/decrement operator
    BoundaryFaceIter operator++(int) {
        BoundaryFaceIter cpy = *this;
        ++(*this);
        return cpy;
    }
    BoundaryFaceIter operator--(int) {
        BoundaryFaceIter cpy = *this;
        --(*this);
        return cpy;
    }
    BoundaryFaceIter operator+(int _n) {
        BoundaryFaceIter cpy = *this;
        for(int i = 0; i < _n; ++i) {
            ++cpy;
        }
        return cpy;
    }
    BoundaryFaceIter operator-(int _n) {
        BoundaryFaceIter cpy = *this;
        for(int i = 0; i < _n; ++i) {
            --cpy;
        }
        return cpy;
    }
    BoundaryFaceIter& operator+=(int _n) {
        for(int i = 0; i < _n; ++i) {
            ++(*this);
        }
        return *this;
    }
    BoundaryFaceIter& operator-=(int _n) {
        for(int i = 0; i < _n; ++i) {
            --(*this);
        }
        return *this;
    }

    BoundaryFaceIter& operator++();
    BoundaryFaceIter& operator--();

private:
    FaceIter bf_it_;
};

//===========================================================================

} // Namespace OpenVolumeMesh

#endif /* ITERATORS_HH_ */
