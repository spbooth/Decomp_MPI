#ifndef RESHAPE
#define RESHAPE
/** @file reshape.h 
 * Data reshape library
 */
#include <mpi.h>
#include "index.h"
/**
 * @private
 */
void raise_error(MPI_Comm comm,char *msg, ...);


typedef struct dim_decomp *DimDecomp;

/** rank_mapper functions map a global coordinate from min_global:max_global to a 
 * processor rank in the range
 * 0:proc_size-1. This usually represents a position along an axis of a processor grid rather than
 * an absolute rank.
 */
typedef int (*rank_mapper)(DimDecomp, int);
/** A offset_mapper function maps a global coordinate from min_global:max_global to a local
 * memory offset. The unit of the offset depends on the context, e.g. the size of the base type and
 * which dimension the DimDecomp is applied to.
 */
typedef int (*offset_mapper)(DimDecomp, int);

/** An optional function to print additional info abut the decomposition
 *
 */
typedef void (*print_decomp)(FILE *,DimDecomp);

/** function to clone the decomp.
 *
 */
typedef DimDecomp (*clone_decomp)(DimDecomp);
/** A DimDecomp represents a decomposition over a dimension. The basic type can be extended
 * to implement different strategies such as block cyclic etc. It is intended that 
 * application code might provide their own decomposition strategies. It should also be possible to use
 * decorator pattern to add additional transformations like cyclic shifts or slice selection.
 *
 * To allow the underlying object to be extended the structures should always be allocated with calloc
 * so additional fields are null/zero
 */
struct dim_decomp{
  rank_mapper rank_offset;  //!< method to convert global coord to rank offset
  offset_mapper type_offset; //!< method to convert global coord to type offset
  void (*destroy)(DimDecomp ); //!< method to clean this type before free, may be null
  print_decomp printer; //!< method to print debug info about the configuration
  clone_decomp cloner; //!< method to clone this decomposition
  char *type;   //!< runtime name of decomposition type for debugging
  int coord_size; //!< Number of coordinates in this dimension.
  int proc_size;  //!< number of processors to decompose over.
};
/** @brief Deallocate a DimDecomp.
 */
void freeDimDecomp(DimDecomp );
/** @brief clone the DimDecomp
 *
 */
DimDecomp cloneDimDecomp(DimDecomp );

/** calculate the length of the coordinate dimension
 * @param d The @ref DimDecomp to query.
 */
int sizeDimDecomp(DimDecomp d);
void describeDimDecomp(FILE *out,DimDecomp d);

typedef struct axis_decomp *AxisDecomp;
/** An AxisDecomp combines a @ref DimDecomp with stride and location information
 * needed to generate offsets and to identify which global coordinates are local
 * to the current processor.
 * Consider this a final type 
 */
struct axis_decomp{
  DimDecomp decomp;  //!< decomposition along this axis
  MPI_Comm comm;    //!< MPI communicator
  int memory_stride; //!< memory stride.
  int rank_stride; //!< rank stride
  int my_rank_in_dimension; //!< position of this processor along the axis.
  int n_local; //!< number of local points
  int max_local; //!< maximum local offset (without stride applied)
  Index index; //!< index of local offsets (without stride applied)

};

/* Make a null terminated array of Index objects corresponsing to
 * a map of local data to each rank in the dimension
 *
 */
Index *makeIndexList(AxisDecomp local,DimDecomp remote);

/** Make the Index corresponding to this processors fragment
 * of the global dataset.
 *
 * This can be used for making a MPI-IO file view
 * @private
 */

Index makeGlobalIndex(AxisDecomp local);
/** Expand up an datatype for this processors fragment of the global dataset.
 * This is for building an MPI-IO file view.
 * @param result output datatype.
 * @param extent extent of base type
 * @param src input datatype
 * @param mine AxisDecomp to expand.
 * @param stride underlying file stride.
 * @return output file stride.
 */
int expandIOType(MPI_Datatype *result,MPI_Aint extent,MPI_Datatype src,AxisDecomp mine,int stride);

/** Create an @ref AxisDecomp
 * @param comm MPI communicator.
 * @param d DimDecomp speficying the decomposition along the axis.
 * @param stride memory srtide along this axis.
 * @param rank_stride process rank stride along this axis.
 */
AxisDecomp makeAxisDecomp(MPI_Comm comm,DimDecomp d,int stride,int rank_stride);
/** Free an AxisDecomp (together with the nested @ref DimDecomp )
 */
void freeAxisDecomp(AxisDecomp);
void describeAxisDecomp(FILE *out,AxisDecomp d);
/** Does this global coordinate (along the specified axis) pass through the local processor.
 * @param d AxisDecomp to be queried.
 8 @param global global coordinate.
 */
int isLocal(AxisDecomp d, int global);

/** descriptor for a multi-dimensional decomposition 
 * The set of AxisDecomp descriptors should be orthogonal and
 * should uniquely identify a single location for each global coordinate.
 * They should also be defined on the same communicator. 
 */
typedef struct decomp *Decomp;
/** Implemenation of a @ref Decomp.
 */
struct decomp{
  int ndim;     //!< number of dimensions
  MPI_Comm comm; //!< MPI communicator
  int proc_used; //!< number of processors used from communicator
  AxisDecomp *dims; //!< array of @ref AxisDecomp defining decomposition
} ;

/** Make a multi-dimensional decomposition
 * @param ndim number of dimensions.
 * @param dims array of @ref AxisDecomp
 */
Decomp makeDecomp(int ndim,AxisDecomp dims[ndim]);

/** Make a multi-dimensional decomposition with default
 * memory layout.
 * @param ndim number of dimensions.
 * @param dims array of @ref DimDecomp
 */
Decomp makePackedDecomp(MPI_Comm comm,int ndim, DimDecomp dims[ndim]);
/** free a @ref Decomp (together with the nested @ref AxisDecomp objects).
 */
void freeDecomp(Decomp d);

void describeDeomp(FILE *out, Decomp d);
/** Get the maximum memory extent of the Decomp
 *
 */
int extent(Decomp d);
/** test to see  if two decompositions are compatible
 * ie can you assign from one to the other.
 * @param a first @ref Decomp
 * @param b second @ref Decomp
 */
int compatible(Decomp a, Decomp b);
#define COMPATIBLE 1
#define NOT_COMPATIBLE 0
/** test to see  if two decompositions are equivalent.
 * @param a first @ref Decomp
 * @param b second @ref Decomp
 */
int equivalent(Decomp a, Decomp b);
#define COMPATIBLE 1
#define NOT_COMPATIBLE 0

int myRank(Decomp);
/** Does a dingle dimension coordinate pass through this processor.
 * @param d The @ref Decomp.
 * @param dim The dimension to query.
 * @param pos The global coordinate to check.
 */
int isCoordLocal(Decomp d, int dim, int pos);
/** Check is a single global coordinate lives on this processor.
 * @param d The @ref Decomp.
 * @param pos an array giving the coordinate to query.
 */
int areCoordsLocal(Decomp d, int *pos);
/** Get the local offset of a local coordinate
 * @param d The @ref Decomp.
 * @param pos an array giving the coordinate to query.
 */
int getOffset(Decomp d, int *pos);


/** A type entry represents a processor rank and an MPI datatype.
 * When fully constructed this will be a send or receive (always length 1)
 * However the same structure is used for intermediate values as the messages are
 * build up dimension by dimension.
 * @private
 */
struct type_entry{
  int rank;
  MPI_Datatype type;
};

/** @private
 */
struct type_entry_list{
  int size;
  int count;
  struct type_entry *list;
};

/**
 * @private
 */
typedef struct type_entry_list *TypeEntryList;

#define INITIAL_TYPELIST_SIZE 16
/** make an initially empty list.
 * @private
 */
TypeEntryList makeTypeEntryList();
/**
 * @private
 */
void commitTypeEntryList(TypeEntryList);

/** Free the list including the MPI datatypes
 * @private
 */
void  freeTypeEntryList(TypeEntryList);
/**
 * @private
 */
void addTypeEntry(TypeEntryList list,int rank, MPI_Datatype type);

/** Expand all entries in the input list generating a new list
 * @private
 */ 
TypeEntryList expandList(int max_proc,MPI_Aint extent,TypeEntryList list,AxisDecomp mine,AxisDecomp remote);

/** print a summary of the TypEntryList
 * @private
 */
void printTypeEntryList(FILE *out, TypeEntryList list);
/** A plan for reshaping data 
 */

struct reshape_plan{
  MPI_Comm comm; //!< MPI communicator
  Decomp src;   //!< source decomposition copy for info only 
  Decomp dest;  //!< destination copies for info only
  int *order; //!< order to expand dimensions
  TypeEntryList sends; //!< list of send operations
  TypeEntryList recvs; //!< list of recv operations
};
typedef struct reshape_plan *ReshapePlan;

/** create a plan for converting data of type base from a src to a dest decomposition.
 * This is a collective call.
 * @param comm MPI communicator.
 * @param base base MPI datatype for the data being moved.
 * @param src source data @ref Decomp.
 * @param dest destination data @ref Decomp.
 */
ReshapePlan makeReshapePlan(MPI_Comm comm,MPI_Datatype base,Decomp src,Decomp dest);

/** free an existing ReshapePlan
 * @param plan plan to free.
 */
void freeReshapePlan(ReshapePlan plan);

/** move data using a pre-defiend plan
 * @param comm MPI communicator
 * @param plan The @ref ReshapePlan to execute.
 * @param src pointer to source data.
 * @param dest pointer to destination data.
 */
void runReshapePlan(MPI_Comm comm,ReshapePlan plan,void *src, void *dest);

/** Check a plan against array sizes.
 * @param plan plan to check
 * @param type_size size of base type.
 * @param src_len length of source array
 * @param dest_len length of dest array.
 */
int checkReshapePlanBounds(ReshapePlan plan, int type_size, int src_len,int dest_len);

/** Make a file-view datatype corresponding to this processors
 * data. Higher/later dimensions change more slowly in file layout
 * @param result datatype
 * @param src datatype
 * @param d Decomp to map.
 */
void makeIOType(MPI_Datatype *result, MPI_Datatype src, Decomp d);
#endif
