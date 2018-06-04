/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_KVSTORE_GPU_TOPOLOGY_H_
#define MXNET_KVSTORE_GPU_TOPOLOGY_H_
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <random>
#include <stack>
#include <unordered_set>
#include <unordered_map>

#define MAX_DEPTH 16

namespace mxnet {
namespace kvstore {

void PrettyPrintTopology(const std::vector<std::vector<size_t>> topo) {
  std::cout << "    ={";
  for (unsigned row = 0; row < topo.size(); ++row) {
    if (row != 0)
      std::cout << "      ";
    std::cout << "{";
    for (unsigned col = 0; col < topo[0].size(); ++col) {
      std::cout << topo[row][col];
      if( col != topo[0].size()-1 )
        std::cout << ", ";
    }
    std::cout << "}";
    if ( row == topo.size()-1 )
      std::cout << "};";
    else
      std::cout << ",";
    std::cout << std::endl;
  }
}

void PrintTopo( const std::string& str, const std::vector<size_t>& topo_row, 
    std::vector<size_t> scan_row ) {
  std::cout << str << ":\n";
  int depth = scan_row.size()-1;
  for (int row = 0; row < depth; ++row) {
    int start = scan_row[row];
    int end   = scan_row[row+1];
    for (; start<end; start++) {
      for (int i = 0; i < (2<<(depth-row-2))+1; ++i) {
        std::cout << " ";
      }
      //std::cout << " ";
      std::cout << topo_row[start];
    }
    std::cout << std::endl;
  }
}

template <typename T>
void PrintMatrix( const std::string& str, const std::vector<T>& matrix, 
    int num_rows, int num_cols ) {
  
  std::cout << str << ":\n";
  int count = 0;
  for (int row = 0; row < num_rows; ++row) {
    for (int col = 0; col < num_cols; ++col) {
      std::cout << matrix[count++] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T>
void PrintVector( const std::string& str, const std::vector<T>& vec ) {
  std::cout << str << ":\n";
  for (unsigned i = 0; i < vec.size(); ++i)
    std::cout << vec[i] << " ";
  std::cout << std::endl;
}

// Get relative performance of NVIDIA GPUs
// 0: Self-connection
// 1: PCI-E
// 2: 1 NVLink connection
// 3: 2 NVLink connections
//
// Generate 2 things:
// 1) adjacency matrix with row/col numbering from 0, 1, ..., n_gpu
// 2) mapping from 0, 1, ..., n_gpu to dev_id
//   -used to map from 0, 1, ..., n_gpu back to dev_id for topology, which will
//    be used by kvstore to do communication
//   -used to build adjacency matrix with 0, 1, ..., n_gpu numbering
template <typename T>
void GetP2PWeight( std::vector<T>&             matrix, 
                   const std::vector<Context>& devs,
                   std::vector<int>&           zero_dev_id,
                   bool                        print=false ) {
  int num_gpus = devs.size();
  int count    = 0;
  for (auto d : devs) {
    zero_dev_id[count] = d.dev_id;
    count++;
  }

  cudaDeviceP2PAttr attr;
  attr = cudaDevP2PAttrPerformanceRank;
  std::vector<int> max(num_gpus, 0);

  for (int row = 0; row < num_gpus; ++row) {
    for (int col = 0; col < num_gpus; ++col) {
      if (row==col) {
        matrix[row*num_gpus+col] = 0;
      } else {
        int value;
        int row_gpu = zero_dev_id[row];
        int col_gpu = zero_dev_id[col];
        cudaDeviceGetP2PAttribute( &value, attr, row_gpu, col_gpu );
        if (value > max[row])
          max[row] = value;
        matrix[row*num_gpus+col] = static_cast<T>(value)+1;
      }
    }
  }

  // Check that all GPUs have at least 1 NVLink connection
  int max_value = 0;
  for (unsigned int i = 0; i < max.size(); ++i) {
    if (max[i] > max_value)
      max_value = max[i];
  }

  // If all GPUs have at least 1 NVLink connection, then we can use NVLink only
  // to communicate instead of going over PCI-E
  if (max_value > 0) {
    for (auto& matrix_value : matrix) {
      matrix_value = (matrix_value==1) ? 0 : matrix_value;
    }
  }
  PrintMatrix( "Weight W", matrix, num_gpus, num_gpus );
}

// Dense matrix-vector multiplication
// Assume: matrix is square
template <typename T>
void gemv( const std::vector<T>& A,
           const std::vector<int>& x,
           std::vector<T>&       y ) {
  int nrows = x.size();
  int count = 0;
  for (int row=0; row<nrows; ++row) {
    y[row] = 0;
    for (int col=0; col<nrows; ++col) {
      y[row] += A[count]*static_cast<T>(x[col]);
      count++;
    }
  }  
}

// Element-wise multiplication between 2 dense vectors
//   w = w * alpha*u
template <typename T>
void ewisemult( const std::vector<int>& u,
                T                       alpha,
                std::vector<T>&         w ) {
  int nelem = u.size();
  for (int i=0; i<nelem; ++i) {
    w[i] *= alpha*static_cast<T>(u[i]);
  }
}

// Element-wise addition between 2 dense vectors
//   w = w + alpha*u
template <typename T>
void ewiseadd( const std::vector<T>& u,
               T                     alpha,
               std::vector<T>&       w ) {
  int nelem = u.size();
  for (int i=0; i<nelem; ++i) {
    w[i] += alpha*static_cast<T>(u[i]);
  }
}

// Computes best 2 nodes a,b to swap given objective function:
//   g = max_{a \in A, b \in B} D(a) + D(b) - 2*W(a,b)
//
// Optimization: Only need to look at upper triangular since weight matrix is
//   symmetric
template <typename T>
void FindBestMove( const std::vector<T>&          W,
                   const std::vector<int>&        P_temp,
                   const std::vector<T>&          D,
                   const std::unordered_set<int>& used,
                   int&                           a,
                   int&                           b,
                   T&                             g ) {
  int nrows = P_temp.size();
  g = 0;
  a = -1;
  b = -1;
  for (int row=0; row<nrows; ++row) {
    if ( P_temp[row]==0 || used.find(row)!=used.end() ) continue;
    for (int col=row+1; col<nrows; ++col) {
      if (P_temp[col]==0 || P_temp[row]==P_temp[col]) continue;

      T cost = D[row]+D[col]-2*W[row*nrows+col];
      if( cost>g ) {
        g = cost;
        a = row;
        b = col;
      }
    }
  }
}

// Performs partition on each existing partition in graph W if partition has
// more than 4 elements in it
// @output: stop returns true if no partitions with >=4 elements found
//               returns false otherwise
template <typename T>
bool KernighanLin( const std::vector<T>&            W,
                   std::vector<int>&                P,
                   int&                             num_partitions,
                   std::vector<std::pair<int,int>>& cluster_pairs,
                   std::mt19937&                    gen ) {

  std::vector<int> histogram(num_partitions, 0);
  std::vector<int> P_temp(P.size(), 0);
  std::vector<int> P_temp2(P.size(), 0);
  std::vector<T>   D(P.size(), 0);
  std::vector<T>   D_temp(P.size(), 0);

  // 0) For every partition, determine if it can be partitioned further.
  //    To do this, we must do a histogram of each partition:
  for (unsigned i=0; i<P.size(); ++i) {
    histogram[P[i]]++;
  }

  bool stop       = true;
  for (unsigned color=0; color<histogram.size(); ++color) {
    int partition_size = histogram[color];
    // Save cluster in preparation for push to topo in GenerateBinaryTree()
    if (partition_size <= 2) {
      cluster_pairs.push_back(std::make_pair<int,int>(
          static_cast<int>(color),-partition_size));

    // Do Kernighan-Lin if clustering is necessary
    } else {
      stop = false;

      // 1) If it has more than 4 elements, we can partition further.
      //    Assign random balanced partition of it
      //   -balanced is more important than random, so allocate first half to A
      //    and rest to B
      int first_partition = 0;
      int target_partition = partition_size/2;
      std::vector<int> cluster_list;

      for (unsigned i = 0; i < P.size(); ++i) {
        // Required to shift from [0,1] to {-1,1}
        //  1 means vertex i is in Cluster A
        // -1 means vertex i is in Cluster B
        if (P[i] == static_cast<int>(color)) {
          cluster_list.push_back(i);
          //std::cout << "Number in Cluster A: " << first_partition << "\n";
          //std::cout << "Put vertex " << i << " in Cluster " << P_temp[i] << "\n";
        } else
          P_temp[i] = 0;
      }

      // 1b) Shuffle using random generator
      std::shuffle(cluster_list.begin(), cluster_list.end(), gen);
      //PrintVector("Partition permutation", cluster_list);
      for (unsigned i = 0; i < cluster_list.size(); ++i) {
        if (first_partition < target_partition) {
          int dest = cluster_list[i];
          P_temp[dest] = 1;
          first_partition++;
        } else {
          int dest = cluster_list[i];
          P_temp[dest] = -1;
        }
      }
      //PrintVector("Partition candidate", P_temp);

      // 2) Do iterations of Kernighan-Lin until convergence
      T   g_max = 0;
      int g_k   = -1;
      unsigned count = 0;
      do {
        count++;
        P_temp2 = P_temp;

        // a) Compute difference between external and internal costs of all 
        //    elements in vector D
        gemv( W, P_temp, D );
        //PrintVector( "D pre-ewisemult", D );
        ewisemult( P_temp, -1.f, D );
        //PrintVector( "D post-ewisemult", D );

        // av and bv are used to hold candidates for moving
        // gv stores the score associated with move
        std::vector<int> av;
        std::vector<int> bv;
        std::vector<T>   gv;

        std::unordered_set<int> used;

        for (int iter=0; iter<partition_size/2; ++iter) {
          // b) Find best move by looking through upper triangular of W matrix
          int a, b;
          T   g;
          FindBestMove( W, P_temp, D, used, a, b, g );
          if (g > 0) {
            //std::cout << "Best move found in iter " << iter;
            //std::cout << ": " << a << " -> " << b << " : " << g << "\n";
          } else {
            //std::cout << "No moves found in iter  " << iter << std::endl;
            g_max = 0;
            break;
          }

          // c) Store best move to av, bv, gv
          av.push_back(a);
          bv.push_back(b);
          gv.push_back(g);

          // d) Eliminate best move from consideration in vector P_temp
          P_temp[a] *= -1;
          P_temp[b] *= -1;
          used.insert(a);
          used.insert(b);

          // e) Update D using P_temp
          //PrintVector( "P_temp post-update", P_temp );
          gemv( W, P_temp, D );
          //PrintVector( "D pre-ewisemult", D );
          ewisemult( P_temp, -1.f, D );
          //PrintVector( "D post-ewisemult", D );
          D[a] = 0;
          D[b] = 0;
          //PrintVector( "D post-ewisemult", D );
        }

        // 3) Find when to stop by doing linear scan through gv
        //    Recompute score g_max
        for (unsigned k = 0; k < gv.size(); ++k) {
          if (k > 0)
            gv[k] += gv[k-1];
          if (gv[k] > g_max) {
            g_max = gv[k];
            g_k   = k + 1;
          }
        }

        // 4) If move is "good", commit moves by updating P_temp and P_temp2
        //    Otherwise, rollback changes to P_temp2
        if (g_max > 0) {
          for (int i = 0; i < g_k; i++) {
            //std::cout << g_max << " " << g_k << " " << i << " " << av.size() << " " << bv.size() << " " << gv.size() << std::endl;
            int a      = av[i];
            int b      = bv[i];
            int temp   = P_temp2[a];
            P_temp2[a] = P_temp2[b];
            P_temp2[b] = temp;

            P_temp     = P_temp2;
          }
        } else {
          P_temp = P_temp2;
        }
      } while (g_max > 0 && count <= P.size());

      // 5) Update P using P_temp
      int moves = 0;
      for (unsigned i=0; i<P.size(); ++i) {
        if (P_temp[i]==-1) {
          P[i] = num_partitions;
          moves++;
        }
      }
      //std::cout << "New color " << num_partitions << " with " << moves;
      //std::cout << " elements\n";
      cluster_pairs.push_back(std::make_pair<int,int>(static_cast<int>(color), 
          static_cast<int>(num_partitions)));

      num_partitions++;
    }
  }

  return stop;
}

// Returns root of a given color if found in roots
// Returns -1 if it is not found
int GetRoot( const std::vector<int>&        P, 
             int                            color, 
             const std::unordered_set<int>& roots ) {
  for (auto root : roots) {
    if (P[root]==color)
      return root;
  }
  return -1;
}

// Returns root of a given color if found in roots
// Returns -1 if it is not found
int GetChild( const std::vector<int>&        P, 
              int                            color, 
              int                            parent ) {
  int size = P.size();
  for (int i = 0; i < size; ++i) {
    //std::cout << "Child " << i << ": " << P[i] << std::endl;
    if (P[i] == color && i != parent)
      return i;
  }
  return -1;
}

// Computes best 2 nodes a,b to swap given objective function:
//   g = max_{a \in A, b \in B} 2*W(a,b)
//
// Optimization: Only need to look at upper triangular since weight matrix is
//   symmetric
template <typename T>
void FindBestEdge( const std::vector<T>&   W,
                   const std::vector<int>& P,
                   int                     parent,
                   int                     dest_cluster,
                   std::vector<int>&       b,
                   T&                      g ) {
  int nrows = P.size();
  int row   = parent;
  g         = 0;
  b.push_back(-1);
  for (int col=0; col<nrows; ++col) {
    if (col==row || P[col]!=dest_cluster) continue;

    T cost = W[row*nrows+col];
    if( cost > g ) {
      b.clear();
    }
    if( cost >= g ) {
      b.push_back(col);
      g = cost;
    }
  }
}

// Given a vector of color pairs, appends to binary tree matrix topo
template <typename T>
int GenerateBinaryTree( std::vector<T>&                  W,
                        const std::vector<int>&          P,
                        std::vector<std::pair<int,int>>& cluster_pairs, 
                        std::unordered_set<int>&         roots,
                        std::vector<size_t>&             topo_row,
                        std::vector<size_t>&             scan_row,
                        std::mt19937&                    gen ) {
  std::unordered_set<int>     new_roots;
  std::unordered_map<int,int> new_topo;
  int reset = 0;

  for (unsigned i = 0; i < cluster_pairs.size(); ++i) {
    //std::cout << "Cluster pair " << i << std::endl;
    if (i==0)
      scan_row.push_back(topo_row.size());
    //std::cout << "Pair " << i << ": " << cluster_pairs[i].first << " " << cluster_pairs[i].second << std::endl;
    int parent, child = -1;
    if (cluster_pairs[i].second==-2) {
      // Root must exist in first element of pair
      int color  = cluster_pairs[i].first;
      parent     = GetRoot( P, color, roots );
      if (parent == -1) return 1;
      child      = GetChild(P, color, parent);
      //std::cout << "Best link (case 1): " << color << ": " << parent << " -> " << child << ": " << std::endl;
    } else if (cluster_pairs[i].second==-1) {
      int color  = cluster_pairs[i].first;
      parent     = GetRoot( P, color, roots );
      if (parent == -1) return 1;
      child      = parent;
      //std::cout << "Best link (case 2): " << color << ": " << parent << " -> " << child << ": " << std::endl;
    } else {
      // Root must exist in either first or second element of pair
      int color  = cluster_pairs[i].first;
      parent     = GetRoot(P, color, roots);
      color      = (parent==-1) ? cluster_pairs[i].second  : color;
      parent     = (parent==-1) ? GetRoot(P, color, roots) : parent;

      int from_cluster = color;
      int dest_cluster = (from_cluster==cluster_pairs[i].first) ? 
          cluster_pairs[i].second : cluster_pairs[i].first;

      std::vector<int> candidates;
      T weight;
      FindBestEdge( W, P, parent, dest_cluster, candidates, weight );

      // If no candidates
      if (candidates[0]!=-1) {
      /*if (candidates[0] == -1) {
        std::cout << "Appending candidates\n";
        candidates.clear();
        for (unsigned col = 0; col < P.size(); ++col) {
          if (W[parent*P.size()+col] > 0)
            for (
            candidates.push_back(col);
          reset = 2;
        }
      }*/
        // Look for candidate that has not been used at this level or previous
        // levels
        /*for (unsigned i = 0; i < candidates.size(); ++i) {
          bool exit = true;
          int last = scan_row.size()-1;
          for (auto it = new_topo.begin(); it != new_topo.end(); ++it) {
            std::cout << "Testing " << candidates[i] << " " << it->second << std::endl;
            if (candidates[i] == it->second) {
              std::cout << candidates[i] << " has been encountered before\n";
              exit = false;
              break;
            }
          }
          if (exit) {
            child = candidates[i];
            std::cout << "GPU " << child << " not found before!\n";
            break;
          }
        }*/
        std::shuffle(candidates.begin(), candidates.end(), gen);
        child = candidates[0];
      }

      if (child == -1) {
        //std::cout << "No path to other cluster found from " << parent << " at level " << scan_row.size() << std::endl;
        new_roots.insert(parent);

        //child = parent;
        return 1;
        /*else {
          child = parent;
          std::cout << "Best link (case 4): " << parent << " -> " << child << ": " << std::endl;
        }*/
      } else {
        //std::cout << "Best link (case 3): " << parent << " -> " << child << ": " << weight << std::endl;
        new_roots.insert(parent);
        new_roots.insert(child);
      }
    }

    new_topo[parent] = child;
    int num_rows = P.size();
  }

  int depth = scan_row.size();
  int start = scan_row[depth-2];
  int end   = scan_row[depth-1];

  for (int i = start; i < end; ++i) {
    int parent = topo_row[i];
    int child;

    // If not first, check previous level whether or not we are encountering 
    // this root for the first time in this level of the tree
    if (i != start && parent == topo_row[i-1])
      child = parent;
    else
      child = new_topo[parent];
    topo_row.push_back(parent);
    topo_row.push_back(child);
    //std::cout << "New pair: " << parent << " " << child << " " << new_topo[parent] << std::endl;
  }

  cluster_pairs.clear();
  roots.clear();
  roots = std::move(new_roots);

  return reset;
}

int ComputeDepth( int n ) {
  for (int depth = 0; depth < MAX_DEPTH; ++depth) {
    int num = 2 << depth;
    if (n <= num)
      return depth+1;
  }
}

template <typename T>
bool IsValid( const std::vector<T>&   W,
              const std::vector<int>& state,
              int                     num_elements,
              int                     row,
              int                     depth ) {

  for (int i = 0; i < depth; ++i) {
    int stride = 1 << i;
    for (unsigned j = 0; j+stride < row; j += 2*stride) {
      int from   = state[j];
      int dest   = state[j+stride];
      //std::cout << "Comparing " << j << " and " << j+stride << " in row " << row << std::endl;
      if (W[from*num_elements + dest] <= static_cast<T>(0) && from != dest) {
        //std::cout << "Not valid: no edge from " << from << " to " << dest << std::endl;
        return false;
      }
    }
  }

  std::unordered_set<int> found;
  std::vector<int>        found_vec(num_elements,0);
  for (auto val : state) {
    if (val == -1)
      continue;
    if (val < num_elements) {
      if (found.find(val) == found.end()) {
        found.insert(val);
        found_vec[val] = 1;
      }
    } else {
      //std::cout << "Not valid: " << val << " exceeds # of GPUs\n";
      return false;
    }
  }
  int modifier = (1 << depth) - num_elements;
  int num_found= found.size();

  if (row < num_elements) {
    if (num_found > row || num_found < row - modifier) {
      //std::cout << "Not valid: " << found.size() << " rows found but expected between " << row << " and " << row - modifier << std::endl;
      return false;
    }
  } else if (row == state.size())
    for (int i = 0; i < num_elements; ++i)
      if (found_vec[i] == 0)
        return false;

  return true;
}

void Postprocess( std::vector<int>& result, int num_elements, int depth) {

  std::vector<int> histogram(num_elements, 0);
  for (unsigned i = 0; i < result.size(); ++i) {
    int val = result[i];
    histogram[val]++;
  }

  for (int i = 0; i == 0; ++i) {
    int stride = 1 << i;
    for (int j = result.size()-1; j-stride >= 0; j -= 2*stride) {
      //std::cout << "Comparing " << j << " and " << j-stride << std::endl;
      int from = result[j];
      int dest = result[j-stride];
      if (histogram[from] > 1 && from != dest) {
        //PrintVector("Old histogram", histogram);
        //std::cout << "Swapping from " << from << " to " << dest << " on indices " << j << " and " << j-stride << std::endl;
        result[j] = dest;
        histogram[from]--;
        //PrintVector("New histogram", histogram);
        //PrintVector("New result", result);
      }
    }
  }  
}

template <typename T>
T GetTreeWeight( const std::vector<T>&   W, 
                 const std::vector<int>& result, 
                 int                     num_elements,
                 int                     depth) {
  T weight = 0.f;
  std::unordered_set<int> links_used;

  for (int i = 0; i < depth; ++i) {
    int stride = 1 << i;
    std::vector<bool> nodes_used(num_elements, false);
    for (unsigned j = 0; j+stride < result.size(); j += 2*stride) {
      int from   = result[j];
      int dest   = result[j+stride];
      if (from != dest) {
        weight += W[from*num_elements+dest];

        // Penalize: (1) use of redundant edges in a single tree
        //           (2) repeated use of a GPU in a single tree at the same 
        //               level above the leaf level
        if (links_used.find(from*num_elements+dest) != links_used.end()) {
          weight -= 100;
          //std::cout << "Penalty 1: " << from << " to " << dest << std::endl;
        }
        links_used.insert(from*num_elements+dest);
        links_used.insert(dest*num_elements+from);
        //std::cout << "Not valid: no edge from " << from << " to " << dest << std::endl;
      }

      nodes_used[from] = true;
      if (i > 0 && nodes_used[dest]) {
        weight -= 10;
        //std::cout << "Penalty 2: " << from << " and " <<  dest << " seen before\n";
      }
      nodes_used[dest] = true;
    }
  }

  return weight;
}

void FormTopology( const std::vector<int>& result, 
                   std::vector<size_t>&    topo_row,
                   std::vector<size_t>&    scan_row,
                   int                     depth ) {
  scan_row.push_back(topo_row.size());
  for (int i = depth; i > 0; --i) {
    int stride = 1 << i;
    for (unsigned j = 0; j < result.size(); j += stride) {
      int from = result[j];
      topo_row.push_back(from);
    }
    scan_row.push_back(topo_row.size());
  }

  // Insert at the end, result vector
  topo_row.insert(topo_row.end(), result.begin(), result.end());
}

template <typename T>
void Backtrack( const std::vector<T>& W,
                std::vector<int>&     state,
                std::vector<int>&     best_result,
                T&                    best_result_weight,
                int                   row,
                int                   num_elements,
                int                   depth ) {
  if (row == state.size()) {
    std::vector<int> result = state;
    Postprocess(result, num_elements, depth);
    T weight = GetTreeWeight(W, result, num_elements, depth);
    if (weight > best_result_weight) {
      std::swap(best_result_weight, weight);
      best_result        = result;
      //std::cout << "New best weight: " << best_result_weight << " > " << weight << std::endl;
      //PrintVector("New best", result);
    } else {
      //std::cout << "Not best weight: " << weight << " < " << best_result_weight << std::endl;
      //PrintVector("Not best", result);
    }
    return;
  }

  for (unsigned j = 0; j < num_elements; ++j) {
    state[row] = j;
    //PrintVector("Trying state", state);
    if (IsValid(W, state, num_elements, row+1, depth)) {
      Backtrack( W, state, best_result, best_result_weight, row+1, num_elements,
          depth );
      state[row] = -1;
    } else
      state[row] = -1;
  }
}

template <typename T>
void UpdateWeight( std::vector<T>&            W,
                   const std::vector<size_t>& topo_row,
                   int                        num_elements,
                   float                      alpha ) {
  for (unsigned i = 1; i < topo_row.size() - 1; i += 2) {
    unsigned parent = topo_row[i];
    unsigned child  = topo_row[i+1];
    if (parent >= num_elements*num_elements || 
        child >= num_elements*num_elements)
      std::cout << "W array out of bounds\n";
    else if (parent != child) {
      W[parent*num_elements+child] *= alpha;
      W[child*num_elements+parent] *= alpha;
    }
  }
}

// Do brute-force backtracking approach if Kernighan-Lin fails to find a binary
// tree of height Log P
// Metrics:
// 1) minimize depth
// 2) maximize edge weight
template <typename T>
void BacktrackingGenerateBinaryTree( std::vector<T>&      W, 
                                     int                  num_elements,
                                     int                  root,
                                     std::vector<size_t>& topo_row, 
                                     std::vector<size_t>& scan_row ) {

  // Clear before starting
  topo_row.clear();
  scan_row.clear();

  // Compute depth
  // 5: 3
  // 6: 3
  // 7: 3
  // 8: 3
  // 9: 4
  int depth = ComputeDepth(num_elements);
  int depth_leaves = 1<<depth;

  // State vector
  // -1 means unplaced
  std::vector<int> state(depth_leaves, -1);
  std::vector<int> result(depth_leaves, -1);
  T result_weight = std::numeric_limits<T>::lowest();

  // Place root and try all combinations
  state[0] = root;
  //PrintVector("state", state);

  Backtrack( W, state, result, result_weight, 1, num_elements, depth );
  FormTopology( result, topo_row, scan_row, depth );
}

template <typename T>
void PartitionGraphFromRoot( std::vector<T>&                   W, 
                             int                               num_elements,
                             int                               root,
                             std::vector<std::vector<size_t>>& topo, 
                             std::vector<std::vector<size_t>>& scan,
                             float                             alpha ) {

  int num_partitions = 1;

  // Initialize partition array to indicate which partition each element belongs
  // to beginning with 0
  std::vector<int> P(num_elements, 0);

  // Initialize vector of pairs that will tell us edges between what 2 clusters
  // we should be looking to build the tree from
  std::vector<std::pair<int,int>> cluster_pairs;

  // Initialize vector of roots that will tell us edges between 
  std::unordered_set<int> roots;
  roots.insert(root);

  // Will be used to obtain a seed for the random number engine
  // RNG: Standard mersenne_twister_engine seeded with rd()
  //     -use 0 for testing (TODO: remove this)
  std::random_device rd;
  std::mt19937 gen(1);
  //std::mt19937 gen(rd());

  // Temporary variables for rewinding
  std::vector<int> P_temp;
  int num_partitions_temp;
  std::unordered_set<int> roots_temp;
  std::vector<size_t> topo_temp;
  std::vector<size_t> scan_temp;

  // Determine number of partition levels
  // If first partition, determine root of maximal spanning tree
  bool stop = false;
  int reset = 1;
  int level = 0;

  bool backtrack = dmlc::GetEnv("MXNET_KVSTORE_BACKTRACK", 1);
  while (!backtrack && (!stop || reset)) {
    if (reset == 1) {
      cluster_pairs.clear();
      P_temp              = P;
      num_partitions_temp = num_partitions;
      roots_temp          = roots;
      topo_temp           = topo[root];
      scan_temp           = scan[root];
    }

    // Run Kernighan-Lin to generate partition
    stop = KernighanLin(W, P_temp, num_partitions_temp, cluster_pairs, gen);
    //PrintVector("New partition", P_temp);

    // Use partitions found and a given root to find best inter-cluster edge for    // each pair of clusters, and returns them as roots of next cluster
    // If reset is true, then rewind back to previous clustering
    reset = GenerateBinaryTree(W, P_temp, cluster_pairs, roots_temp, 
        topo_temp, scan_temp, gen);

    if (reset)
      level++;
    if (level > 10) break;
  }

  if (reset == 1) {
    if (!backtrack)
      std::cout << "No valid binary tree found from root " << root << ", try backtracking\n";
    //std::cout << "Trying backtracking\n";
    BacktrackingGenerateBinaryTree(W, num_elements, root, topo[root], 
        scan[root]);
  } else {
    topo[root]     = topo_temp;
    scan[root]     = scan_temp;
  }
  UpdateWeight( W, topo[root], num_elements, alpha );
}

// Generalization from num_elements to list of devices done using zero_dev_id
// mapping, which gets us from 0, 1, ..., n_gpus to dev_id
template <typename T>
void PartitionGraph( const std::vector<T>&             W, 
                     int                               num_elements,
                     const std::vector<int>&           zero_dev_id,
                     std::vector<std::vector<size_t>>& topo, 
                     std::vector<std::vector<size_t>>& scan,
                     float                             alpha=0.7 ) {
  std::vector<T> W_copy = W;

  topo.clear();
  scan.clear();
  for (int i = 0; i < num_elements; ++i) {
    topo.push_back(std::vector<size_t>());
    scan.push_back(std::vector<size_t>());
    topo[i].push_back(i);
    scan[i].push_back(0);
    PartitionGraphFromRoot(W_copy, num_elements, i, topo, scan, alpha);
    scan[i].push_back(topo[i].size());
  }

  // Note: must sum up adj matrix to show link usage before we readjust topo
  // from 0, 1, ..., n_gpus format to dev_id format, which will cause segfault
  std::vector<int> adj(W.size(), 0);
  for (int row = 0; row < num_elements; ++row) {
    for (unsigned col = 1; col < topo[0].size(); col += 2) {
      int from = std::min(topo[row][col], topo[row][col+1]);
      int dest = std::max(topo[row][col], topo[row][col+1]);
      if (from != dest) {
        adj[from*num_elements+dest] += 1;
        adj[dest*num_elements+from] += 1;
      }
    }
  }

  std::vector<std::vector<size_t>> topo_temp(num_elements,
      std::vector<size_t>());
  for (int i = 0; i < num_elements; ++i) {
    for (unsigned j = 0; j < topo[i].size(); ++j) {
      int val = topo[i][j];
      topo_temp[i].push_back( zero_dev_id[val] );
    }
    PrintTopo("Topo_temp", topo_temp[i], scan[i]);
  }

  PrintMatrix("Links", adj, num_elements, num_elements);
  bool backtrack = dmlc::GetEnv("MXNET_KVSTORE_BACKTRACK", 1);
  if (backtrack)
    LOG(WARNING) << "Using Backtracking to generate trees";
  else
    LOG(WARNING) << "Using Kernighan-Lin to generate trees";
}

}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_GPU_TOPOLOGY_H
