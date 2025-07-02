#define TBB_USE_DEBUG 1
#define _CRT_SECURE_NO_WARNINGS
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <time.h>
#include <cstdio>
#include <cassert>
#include <stdio.h>
#include <numeric>
#include <set>
#include <unordered_set>
#include <map>
#include <sstream>
#include <omp.h>
#include <iomanip>
#include <chrono>
#include <thrust/sort.h>
#include "cuda_runtime.h"
#include<thrust/reduce.h>
#include<thrust/device_ptr.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/copy.h>
#include<thrust/execution_policy.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/random_device.hpp>
#include <boost/variant.hpp>
#include <tbb/atomic.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_vector.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_do.h>
#include <tbb/parallel_for_each.h>
#include <tbb/tick_count.h>
#include <tbb/task_scheduler_init.h>

#define shareMemorySizeInBlock 128
#define hIndex 2048
#define _CRT_SECURE_NO_WARNINGS


using namespace std;
//using namespace boost::random;

template<typename K, typename V>
using ConcurrentMap = tbb::concurrent_hash_map<K,V>;

template<typename V>
using cuset = tbb::concurrent_unordered_set<V>;


template<typename K, typename V>
using UMap = std::unordered_map<K,V>;



#define SZ(x) ((int)x.size())
#define ll long long
#define ull unsigned long long
#define ld long double
#define eps 1e-11
#define bucketNum 10


char input_address[2000], output_address [2000] ;

clock_t allStart,allEnd,tStart,tEnd;

set <vector<int>> edges;
map<pair<int,int>,int> edge_sign;
vector < pair <int, int> > list_of_edges;
map < int, int > vertices [2];
vector <int> index_map;
vector <int> vertices_in_left;
vector <int> vertices_in_right;
vector < vector <int> > adj;
vector < vector < int > > sampled_adj_list;
vector <bool> visited;
vector <int> list_of_vertices;
vector <int> vertex_counter;

ll count_wedge;
ll n_vertices;
ll n_edges;
ld exact_n_bf;
ld exact_n_bf_signed;
ld exact_n_bf_unsigned;
ll n_wedge_in_partition[2];
ll largest_index_in_partition[2];

vector <int> clr;
vector <int> hashmap_C;
vector <ll> sum_wedges;
vector <ll> sum_deg_neighbors;
vector <int> aux_array_two_neighboorhood;

std::vector<int> cpu_vertices;
std::vector<int> gpu_vertices;
std::vector<int> gpu_nonZeroRow_vec;
std::vector<int> gpu_nonZeroRow; 




struct edge{
    unsigned int src;
    unsigned int dst;
    int sign;
};

struct cmpStruc{
    __device__ bool operator () (const edge &a, const edge &b)
    {
        return (a.src < b.src) || (a.src == b.src && a.dst < b.dst);
    }
};

class edgeVector{
public:
        unsigned int capcity;
        unsigned int esize;
        edge *Edges;
        edgeVector(){esize = 0; capcity = 0;}
        void init(unsigned int s) { Edges = new edge [s]; capcity = s; return ;}
        void addEdge(int _src,int _dst){
                if(esize >= capcity) {
                        capcity *= 2;
                        edge* tmpEdges = new edge [capcity];
                        memcpy(tmpEdges,Edges,sizeof(edge)*esize);
                        delete [] Edges;
                        Edges = tmpEdges;
                }
                Edges[esize].src = _src;
                Edges[esize].dst = _dst;
                esize ++;
        }
        void clear() {delete [] Edges; return ;}
};

int *edgeOffset;
int *edgeRow;
int *edgeSign; //sign
int *edgeOffset_left;
int *edgeRow_left;
int *adjLength;
int *edgeOffset_right;
int *edgeRow_right;
int *nonZeroRow;
edge *Edges;
clock_t start_, end_;



void add_vertex(int A, int side) {
	if (vertices[side].find(A) == vertices[side].end()) {
		if (side == 0) vertices_in_left.push_back(A);
		else vertices_in_right.push_back(A);
		vertices[side][A] = 1;
	}
}

void get_index(int &A, int side) {
	if (vertices[side].find(A) == vertices[side].end()) {
		vertices[side][A] = largest_index_in_partition[side] ++ ;
	}
	A = vertices[side][A];
}

void add_edge(int &A, int &B, int &sign) {
	add_vertex(A, 0);
	add_vertex(B, 1);
	if (edges.find({A,B,sign}) == edges.end()) {
		edges.insert({A,B,sign});
		n_edges++;
	}
}

bool all_num(string &s) {
	for (int i = 0; i < SZ(s); i++) if ((s[i] >= '0' && s [i] <= '9') == false) return false;
	return true;
}

void resize_all() {
	clr.resize(n_vertices);
	hashmap_C.resize(n_vertices);
	aux_array_two_neighboorhood.resize(n_vertices);
	sum_wedges.resize(n_vertices);
	visited.resize(n_vertices);
	index_map.resize(n_vertices);
	sum_deg_neighbors.resize(n_vertices);
}


void clear_everything() {
	largest_index_in_partition[0] = largest_index_in_partition[1] = 0;
	n_vertices = 0;
	n_edges = 0;
	edges.clear();
	vertices[0].clear(); vertices[1].clear();
	index_map.clear();
	vertices_in_left.clear();
	vertices_in_right.clear();
	adj.clear();
	sampled_adj_list.clear();
	visited.clear();
	list_of_edges.clear();
	vertex_counter.clear();
	clr.clear();
	hashmap_C.clear();
	sum_wedges.clear();
	sum_deg_neighbors.clear();
	aux_array_two_neighboorhood.clear();
}

bool preProcess(unsigned &_nonZeroSize, unsigned int &nodeoffset, unsigned int &nodeNum_end, unsigned int &nodeNum, unsigned int &edgeNum) {
    if (!freopen(input_address, "r", stdin)) {
        perror("freopen failed");
        return false;
    }

    string s;
    cin.clear();
    int size = 0;
    while (getline(cin, s)) {
        stringstream ss(s);
        vector<string> vec_str;
        for (string z; ss >> z; vec_str.push_back(z));
        if (vec_str.size() >= 2) {
            bool is_all_num = true;
            for (int i = 0; i < min(2, (int)vec_str.size()); i++)
                is_all_num &= all_num(vec_str[i]);
            if (is_all_num) {
                int A, B, sign;
                stringstream(vec_str[0]) >> A;
                stringstream(vec_str[1]) >> B;
                stringstream(vec_str[2]) >> sign;
                add_edge(A, B, sign);
                size++;
            }
        }
    }

    cout << "phase : 1 finished------------   File Reading Done -------------------" << endl;

    vertices[0].clear();
    vertices[1].clear();

    largest_index_in_partition[0] = 0;
    largest_index_in_partition[1] = vertices_in_left.size();

    n_vertices = vertices_in_left.size() + vertices_in_right.size();
    adj.resize(n_vertices, vector<int>());

    edgeNum = size;
    nodeNum = n_vertices;

    // ✅ Choose the smaller side for processing
    bool use_left = (vertices_in_left.size() <= vertices_in_right.size());
    int small_side_size = use_left ? vertices_in_left.size() : vertices_in_right.size();
    nodeNum_end = small_side_size;
    nodeoffset = use_left ? 0 : vertices_in_left.size();

    Edges = new edge[edgeNum];
    size = 0;

    for (auto edge : edges) {
        int A = edge[0];
        int B = edge[1];
        int sign = edge[2];

        Edges[size].src = A;
        Edges[size].dst = B;

        get_index(A, 0);
        get_index(B, 1);

        adj[A].push_back(B);
        adj[B].push_back(A);

        edge_sign[{A, B}] = sign;
        edge_sign[{B, A}] = sign;

        size++;
    }

    adjLength = new int[nodeNum];
    edgeOffset = new int[nodeNum + 1];
    edgeRow = new int[edgeNum * 2];
    edgeSign = new int[edgeNum * 2];
    edgeOffset[0] = 0;

    cout << "----------------    phase two finished    --------------- " << endl;

    unsigned int nodePos = 0;
    for (int i = 0; i < nodeNum; i++) {
        edgeOffset[i + 1] = edgeOffset[i] + adj[i].size();
        for (auto &entry : adj[i]) {
            edgeRow[nodePos] = entry;
            edgeSign[nodePos] = edge_sign[{i, entry}];
            nodePos++;
        }
    }

    resize_all();

    cout << "phase three finished" << endl;

    cout << "Number of vertices in Left Partition  = " << vertices_in_left.size() << endl;
    cout << "Number of vertices in Right Partition = " << vertices_in_right.size() << endl;

    // ✅ Begin CPU-GPU split on smaller side
    cpu_vertices.clear();
    gpu_vertices.clear();

    const long long WEDGE_THRESHOLD = 1000;

    int side_start = use_left ? 0 : vertices_in_left.size();
    int side_end = side_start + small_side_size;

    for (int i = side_start; i < side_end; i++) {
        sort(adj[i].begin(), adj[i].end());
        adjLength[i] = adj[i].size();

        long long deg = adj[i].size();
        long long wedge = (deg * (deg - 1)) / 2;

        if (wedge < WEDGE_THRESHOLD)
            cpu_vertices.push_back(i);
        else
            gpu_vertices.push_back(i);

        sum_deg_neighbors[i] = 0;
        for (auto neighbor : adj[i])
            sum_deg_neighbors[i] += adj[neighbor].size();
    }

    cerr << "Processed side: " << (use_left ? "Left" : "Right") << endl;
    cerr << "CPU vertices = " << cpu_vertices.size() << ", GPU vertices = " << gpu_vertices.size() << endl;

    // Debug print
    cout << "List of CPU vertices: ";
    for (auto v : cpu_vertices) cout << v << " ";
    cout << endl;

    cout << "List of GPU vertices: ";
    for (auto v : gpu_vertices) cout << v << " ";
    cout << endl;

    // ✅ Build nonZeroRow and gpu_nonZeroRow
    unsigned int nonZerosize = 0;
    nonZeroRow = new int[nodeNum];
    for (int i = side_start; i < side_end; i++) {
        if (edgeOffset[i] != edgeOffset[i + 1])
            nonZeroRow[nonZerosize++] = i;
    }

    std::unordered_set<int> gpu_vertex_set(gpu_vertices.begin(), gpu_vertices.end());
    gpu_nonZeroRow_vec.clear();
    for (int i = 0; i < nonZerosize; i++) {
        int v = nonZeroRow[i];
        if (gpu_vertex_set.count(v))
            gpu_nonZeroRow_vec.push_back(v);
    }
    gpu_nonZeroRow = gpu_nonZeroRow_vec;

    // Finalize
    int *tmpNonZeroRow = new int[nonZerosize];
    memcpy(tmpNonZeroRow, nonZeroRow, sizeof(int) * nonZerosize);
    delete[] nonZeroRow;
    nonZeroRow = tmpNonZeroRow;
    _nonZeroSize = nonZerosize;

    cout << "The nonZeroSize is " << nonZerosize << ", node_end is " << nodeNum_end << endl;
    for (int j = 0; j < min(10U, _nonZeroSize); j++) cout << nonZeroRow[j] << " ";
    cout << endl;

    fclose(stdin);
    return true;
}


// bool preProcess(unsigned &_nonZeroSize, unsigned int &nodeoffset,unsigned int &nodeNum_end,unsigned int &nodeNum, unsigned int &edgeNum){
//     if (!freopen(input_address, "r", stdin)) {
//         perror("freopen failed");
//         return false;
//     }
//     string s;
//     cin.clear();
//     int size = 0;
//     while (getline(cin, s)) {
//         stringstream ss; ss << s;
//         vector <string> vec_str;
//         for (string z; ss >> z; vec_str.push_back(z));
//         if (vec_str.size() >= 2) {
//             bool is_all_num = true;
//             for (int i = 0; i < min(2, (int)vec_str.size()); i++)
//                 is_all_num &= all_num(vec_str[i]);
//             if (is_all_num) {
//                 int A, B, sign;
//                 ss.clear(); ss << vec_str[0]; ss >> A;
//                 ss.clear(); ss << vec_str[1]; ss >> B;
//                 ss.clear(); ss << vec_str[2]; ss >> sign;
//                 add_edge(A, B, sign);
//                 size++;
//             }
//         }
//     }

//     cout << "phase : 1 finished------------   File Reading Done -------------------" << endl;

//     vertices[0].clear();
//     vertices[1].clear();

//     largest_index_in_partition[0] = 0;
//     largest_index_in_partition[1] = vertices_in_left.size();
//     n_vertices = vertices_in_left.size() + vertices_in_right.size();
//     adj.resize(n_vertices, vector<int>());

//     edgeNum = size;
//     nodeNum = n_vertices;
//     nodeNum_end = (vertices_in_left.size() > vertices_in_right.size()) ? nodeNum : vertices_in_left.size();
//     Edges = new edge[edgeNum];
//     nodeoffset = (vertices_in_left.size() > vertices_in_right.size()) ? vertices_in_left.size() : 0;

//     size = 0;
//     for (auto edge : edges) {
//         int A = edge[0];
//         int B = edge[1];
//         int sign = edge[2];

//         Edges[size].src = A;
//         Edges[size].dst = B;

//         get_index(A, 0);
//         get_index(B, 1);

//         adj[A].push_back(B);
//         adj[B].push_back(A);

//         edge_sign[{A, B}] = sign;
//         edge_sign[{B, A}] = sign;

//         size++;
//     }

//     adjLength = new int[nodeNum];
//     edgeOffset = new int[nodeNum + 1];
//     edgeRow = new int[edgeNum * 2];
//     edgeSign = new int[edgeNum * 2];
//     edgeOffset[0] = 0;

//     cout << "----------------    phase two finished    --------------- " << endl;

//     unsigned int nodePos = 0;
//     for (int i = 0; i < nodeNum; i++) {
//         edgeOffset[i + 1] = edgeOffset[i] + adj[i].size();
//         for (auto &entry : adj[i]) {
//             edgeRow[nodePos] = entry;
//             edgeSign[nodePos] = edge_sign[{i, entry}];
//             nodePos++;
//         }
//     }

//     resize_all();

//     cout << "phase three finished" << endl;

//     // Wedge count partitions
//     const long long WEDGE_THRESHOLD = 1000;
//     cpu_vertices.clear();
//     gpu_vertices.clear();

//     for (int i = 0; i < nodeNum_end; i++) {
//         sort(adj[i].begin(), adj[i].end());
//         adjLength[i] = adj[i].size();
//         long long deg = adj[i].size();
//         long long wedge = (deg * (deg - 1)) / 2;

//         if (wedge < WEDGE_THRESHOLD) {
//             cpu_vertices.push_back(i);
//         } else {
//             gpu_vertices.push_back(i);
//         }

//         sum_deg_neighbors[i] = 0;
//         for (auto neighbor : adj[i]) {
//             sum_deg_neighbors[i] += adj[neighbor].size();
//         }
//     }

//     cerr << "CPU vertices = " << cpu_vertices.size() << ", GPU vertices = " << gpu_vertices.size() << endl;

//     cout << "List of CPU vertices: ";
//     for (auto v : cpu_vertices) {
//         cout << v << " ";
//     }
//     cout << endl;

//     cout << "List of GPU vertices: ";
//     for (auto v : gpu_vertices) {
//         cout << v << " ";
//     }
//     cout << endl;

//     unsigned int nonZerosize = 0;
//     nonZeroRow = new int[nodeNum];
//     for (int i = nodeoffset; i < nodeNum_end; i++) {
//         if (edgeOffset[i] != edgeOffset[i + 1]) {
//             nonZeroRow[nonZerosize++] = i;
//         }
//     }

//     // Extract GPU-specific non-zero rows
//     std::unordered_set<int> gpu_vertex_set(gpu_vertices.begin(), gpu_vertices.end());
//     gpu_nonZeroRow_vec.clear();
//     for (int i = 0; i < nonZerosize; i++) {
//         int v = nonZeroRow[i];
//         if (gpu_vertex_set.count(v)) {
//             gpu_nonZeroRow_vec.push_back(v);
//         }
//     }

//     // Copy to global variable
//     gpu_nonZeroRow = gpu_nonZeroRow_vec;

//     int *tmpNonZeroRow = new int[nonZerosize];
//     memcpy(tmpNonZeroRow, nonZeroRow, sizeof(int) * nonZerosize);
//     delete[] nonZeroRow;
//     nonZeroRow = tmpNonZeroRow;
//     _nonZeroSize = nonZerosize;

//     cout << "The nonZeroSize is " << nonZerosize << ", node_end is " << nodeNum_end << endl;
//     for (int j = 0; j < min(10U, _nonZeroSize); j++)
//         cout << nonZeroRow[j] << " ";
//     cout << endl;

//     fclose(stdin);
//     return true;
// }


__constant__ edge *c_Edges;
__constant__ unsigned int *c_adjLen;
__constant__ unsigned int *c_offset;
__constant__ unsigned int *c_row;
__constant__ int c_threadsPerEdge;
__constant__ long *c_sums;
__constant__ unsigned int * c_bitmap;
__device__ unsigned int *c_bitmap_0;
__device__ unsigned int *c_bitmap_1;
__device__ unsigned int *c_bitmap_2;



__constant__ unsigned int * c_nonZeroRow;
__constant__ unsigned int c_edgeSize;
__constant__ unsigned int c_nodeSize;
__constant__ unsigned int c_nodeoffset;

__device__ int* c_sign;

unsigned int nodeOffset;
unsigned int nonZeroSize;
unsigned int nodeNum;
unsigned int nodeNum_end;
unsigned int edgeNum;
unsigned int direction;





void read_the_graph() {
	clear_everything();
	cerr << " Insert the input (bipartite network) file location" << endl;
	cerr << " >>> "; cin >> input_address;

	cerr << " Processing the graph ... (please wait) \n";


	preProcess(nonZeroSize, nodeOffset, nodeNum_end, nodeNum, edgeNum);

	cerr << " -------------------------------------------------------------------------- \n";
	cerr << " The graph is processed - there are " << nodeNum << " vertices and " << edgeNum << " edges  \n";
	cerr << " -------------------------------------------------------------------------- \n";
}


ll balanced_butterfly_counting_bucketing(const std::vector<int>& cpu_vertices) {
	
	tbb::atomic<ll> balanced_bf_count = 0;
	tbb::atomic<ll> *sign_sum = new tbb::atomic<ll>[nodeNum_end];


    tbb::parallel_for(tbb::blocked_range<int>(0, nodeNum_end), [&](tbb::blocked_range<int> r) {
        for (int u = r.begin(); u < r.end(); ++u) {
              //  fprintf(stderr, "Processing vertex u = %d\n", u);
            
            ConcurrentMap<int, int> count_wedge_with_signs_0;
            ConcurrentMap<int, int> count_wedge_with_signs_1;
            ConcurrentMap<int, int> count_wedge_with_signs_2;
            cuset<int> n_w;

				tbb::parallel_for(tbb::blocked_range<int>(0,SZ(adj[u])),
						[&](tbb::blocked_range<int> r1){
							for(int j = r1.begin(); j < r1.end(); j++)
							{
                                  //fprintf(stderr, "Processing neighbor adj[%d][%d] = %d\n", u, j, adj[u][j]);
     

								if(SZ(adj[adj[u][j]]) < SZ(adj[u]) || ((SZ(adj[adj[u][j]]) == SZ(adj[u])) &&(adj[u][j]<u))){

								tbb::parallel_for(tbb::blocked_range<int>(0,SZ(adj[adj[u][j]])),
										[&](tbb::blocked_range<int> r2){

											for(int k=r2.begin(); k < r2.end(); k++){
                                                        

												if(SZ(adj[adj[adj[u][j]][k]]) < SZ(adj[u]) ||  ((SZ(adj[adj[adj[u][j]][k]]) == SZ(adj[u])) &&(adj[adj[u][j]][k]<u))) {
													tbb::atomic<int> sign_sum = edge_sign[{adj[u][j],adj[adj[u][j]][k]}] + edge_sign[{u,adj[u][j]}];

													n_w.insert(adj[adj[u][j]][k]);

                					        					if(sign_sum == 0) {ConcurrentMap<int, int>::accessor ac; count_wedge_with_signs_0.insert(ac, adj[adj[u][j]][k]); ac->second +=1;}
                					        					else if(sign_sum == 1) {ConcurrentMap<int, int>::accessor ac; count_wedge_with_signs_1.insert(ac, adj[adj[u][j]][k]); ac->second +=1;}
                					        					else if(sign_sum == 2) {ConcurrentMap<int, int>::accessor ac;count_wedge_with_signs_2.insert(ac, adj[adj[u][j]][k]); ac->second +=1;}

												}
											}
										});
								}
							}
						}
						);
        			if(n_w.size() > 0){
					tbb::parallel_for_each(n_w.begin(), n_w.end(),
							[&](int v){

							ConcurrentMap<int, int>::accessor a1;
							ConcurrentMap<int, int>::accessor a2;
							ConcurrentMap<int, int>::accessor a3;
							count_wedge_with_signs_0.find(a1, v);
							count_wedge_with_signs_1.find(a2, v);
							count_wedge_with_signs_2.find(a3, v);
							if(count_wedge_with_signs_0.find(a1, v) && count_wedge_with_signs_2.find(a3, v))
        				    			balanced_bf_count.fetch_and_add(((a1->second+a3->second) * (a1->second+a3->second -1)) / 2);
							else if(count_wedge_with_signs_0.find(a1, v) && !count_wedge_with_signs_2.find(a3, v))
        				    			balanced_bf_count.fetch_and_add(((a1->second) * (a1->second -1)) / 2);
							else if(!count_wedge_with_signs_0.find(a1, v) && count_wedge_with_signs_2.find(a3, v))
        				    			balanced_bf_count.fetch_and_add(((a3->second) * (a3->second -1)) / 2);
							if(count_wedge_with_signs_1.find(a2, v))
								balanced_bf_count += ((a2->second) * (a2->second -1)) / 2;


							});
				}
				else{
					for(auto i : n_w)
					{
						ConcurrentMap<int, int>::accessor a1;
						ConcurrentMap<int, int>::accessor a2;
						ConcurrentMap<int, int>::accessor a3;
						count_wedge_with_signs_0.find(a1, i);
						count_wedge_with_signs_1.find(a2, i);
						count_wedge_with_signs_2.find(a3, i);

							if(count_wedge_with_signs_0.find(a1, i) && count_wedge_with_signs_2.find(a3, i))
        				    			balanced_bf_count.fetch_and_add(((a1->second+a3->second) * (a1->second+a3->second -1)) / 2);
							else if(count_wedge_with_signs_0.find(a1, i) && !count_wedge_with_signs_2.find(a3, i))
        				    			balanced_bf_count.fetch_and_add(((a1->second) * (a1->second -1)) / 2);
							else if(!count_wedge_with_signs_0.find(a1, i) && count_wedge_with_signs_2.find(a3, i))
        				    			balanced_bf_count.fetch_and_add(((a3->second) * (a3->second -1)) / 2);
							if(count_wedge_with_signs_1.find(a2, i))
								balanced_bf_count += ((a2->second) * (a2->second -1)) / 2;



					}
				}

			}
			});

	return balanced_bf_count;
}




__global__ void BFCbyHash2(unsigned int totalNodeNum, unsigned int nodenum_select, int* gpu_nonZeroRow, int gpu_nonZeroSize) {

    int curRowNum = blockIdx.x;
    int lane_id = threadIdx.x;
    long butterfly = 0;
    long sum = 0;

    unsigned int intSizePerBitmap = nodenum_select;
    __shared__ unsigned int sh_bitmap[shareMemorySizeInBlock / 2];

    unsigned int* myBitmap_0 = c_bitmap_0 + blockIdx.x * intSizePerBitmap;
    unsigned int* myBitmap_1 = c_bitmap_1 + blockIdx.x * intSizePerBitmap;
    unsigned int* myBitmap_2 = c_bitmap_2 + blockIdx.x * intSizePerBitmap;

    while (1) {
        int u = (curRowNum < gpu_nonZeroSize) ? gpu_nonZeroRow[curRowNum] : totalNodeNum;
        if (u >= totalNodeNum) {
            break;
        }
        unsigned int* curNodeNbr = c_row + c_offset[u];
        unsigned int curNodeNbrLength = c_offset[u + 1] - c_offset[u];

        if (threadIdx.x == 0) {
            memset(myBitmap_0, 0, sizeof(int) * intSizePerBitmap);
            memset(myBitmap_1, 0, sizeof(int) * intSizePerBitmap);
            memset(myBitmap_2, 0, sizeof(int) * intSizePerBitmap);
            memset(sh_bitmap, 0, sizeof(unsigned int) * shareMemorySizeInBlock / 2);
        }

        for (int i = lane_id; i < nodenum_select; i += blockDim.x) {
            myBitmap_0[i] = 0;
            myBitmap_1[i] = 0;
            myBitmap_2[i] = 0;
        }
        __syncthreads();

        if (gpu_nonZeroSize > 32) {
            for (int i = 0; i < (curNodeNbrLength + blockDim.x - 1) / blockDim.x; i++) {
                int curIndex = i * blockDim.x + threadIdx.x;
                if (curIndex < curNodeNbrLength) {
                    int v = curNodeNbr[curIndex];
                    unsigned int sign_uv = 0;
                    unsigned int start_uv = c_offset[u];
                    unsigned int end_uv = c_offset[u + 1];
                    for (unsigned int k = start_uv; k < end_uv; k++) {
                        if (c_row[k] == v) {
                            sign_uv = c_sign[k];
                            break;
                        }
                    }
                    unsigned int* twoHoopNbr = c_row + c_offset[v];
                    unsigned int twoHoopNbrLength = c_offset[v + 1] - c_offset[v];
                    for (int j = 0; j < twoHoopNbrLength; j++) {
                        unsigned int w = twoHoopNbr[j];
                        if (w > u) {
                            unsigned int sign_vw = 0;
                            unsigned int start_vw = c_offset[v];
                            unsigned int end_vw = c_offset[v + 1];
                            for (unsigned int k = start_vw; k < end_vw; k++) {
                                if (c_row[k] == w) {
                                    sign_vw = c_sign[k];
                                    break;
                                }
                            }
                            int sign_sum = sign_uv + sign_vw;
                            if (sign_sum == 0) {
                                atomicAdd(myBitmap_0 + w - c_nodeoffset, 1);
                            } else if (sign_sum == 1) {
                                atomicAdd(myBitmap_1 + w - c_nodeoffset, 1);
                            } else if (sign_sum == 2) {
                                atomicAdd(myBitmap_2 + w - c_nodeoffset, 1);
                            }
                        }
                    }
                }
            }
        } else {
            for (int i = 0; i < (curNodeNbrLength + blockDim.x - 1) / blockDim.x; i++) {
                int curIndex = i * blockDim.x + threadIdx.x;
                if (curIndex < curNodeNbrLength) {
                    int v = curNodeNbr[curIndex];
                    unsigned int sign_uv = 0;
                    unsigned int start_uv = c_offset[u];
                    unsigned int end_uv = c_offset[u + 1];
                    for (unsigned int k = start_uv; k < end_uv; k++) {
                        if (c_row[k] == v) {
                            sign_uv = c_sign[k];
                            break;
                        }
                    }
                    unsigned int* twoHoopNbr = c_row + c_offset[v];
                    unsigned int twoHoopNbrLength = c_offset[v + 1] - c_offset[v];
                    for (int j = 0; j < twoHoopNbrLength; j++) {
                        unsigned int w = twoHoopNbr[j];
                        if (w > u) {
                            unsigned int sign_vw = 0;
                            unsigned int start_vw = c_offset[v];
                            unsigned int end_vw = c_offset[v + 1];
                            for (unsigned int k = start_vw; k < end_vw; k++) {
                                if (c_row[k] == w) {
                                    sign_vw = c_sign[k];
                                    break;
                                }
                            }
                            int sign_sum = sign_uv + sign_vw;
                            if (sign_sum == 0) {
                                atomicAdd(myBitmap_0 + w - c_nodeoffset, 1);
                            } else if (sign_sum == 1) {
                                atomicAdd(myBitmap_1 + w - c_nodeoffset, 1);
                            } else if (sign_sum == 2) {
                                atomicAdd(myBitmap_2 + w - c_nodeoffset, 1);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        curRowNum += gridDim.x;
    }

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (blockIdx.x < gpu_nonZeroSize) {
        for (int i = 0; i < (nodenum_select + blockDim.x - 1) / blockDim.x; i++) {
            int j = i * blockDim.x + threadIdx.x;
            if (j < nodenum_select) {
                unsigned int two_zeros = myBitmap_0[j]; // - -
                unsigned int one_zero = myBitmap_1[j];  // - +
                unsigned int two_ones = myBitmap_2[j];  // + +

                unsigned int balanced = two_zeros + two_ones;

                if (balanced > 1)
                    sum += (((balanced) * (balanced - 1)) / 2);

                if (one_zero > 1)
                    sum += (((one_zero) * (one_zero - 1)) / 2);
            }
        }
        __threadfence();
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
        if (threadIdx.x % 32 == 0) {
            c_sums[idx >> 5] = sum;
        }
    }
    return;
}





ll my_exact_hashmap_bfc(edge *d_Edges, int *d_adjLen, int *d_edgeOffset, int *d_edgeRow, int *d_nonZeroRow) {
    int curThreadsPerEdge = 1;
    int blockSize = 32;
    int blockNum = 30 * 2048 / blockSize;
    int bitPerInt = sizeof(int) * 6;

    unsigned int intSizePerBitmap = nodeNum;
    unsigned int maxWarpPerGrid = blockNum * blockSize / 32;
    int totalBlocks = blockNum;
    int nodenum_select = nodeNum;

    if ((blockNum * intSizePerBitmap * sizeof(int)) / 1024 > 6 * 1024 * 1024) {
        std::cout << "RUN OUT OF GLOBAL MEMORY!!" << std::endl;
        return 0;
    }

    // === GPU Non-Zero Row Handling ===
    int gpu_nonZeroSize = gpu_nonZeroRow.size();
    int *d_gpu_nonZeroRow;
    cudaMalloc(&d_gpu_nonZeroRow, sizeof(int) * gpu_nonZeroSize);
    cudaMemcpy(d_gpu_nonZeroRow, gpu_nonZeroRow.data(), sizeof(int) * gpu_nonZeroSize, cudaMemcpyHostToDevice);

    // === Allocate and copy graph structure ===
    unsigned int *t_edgeOffset, *t_edgeRow;
    int *t_edgeSign;
    long *t_sum;

    cudaMalloc(&t_edgeOffset, sizeof(unsigned int) * (nodeNum + 1));
    cudaMalloc(&t_edgeRow, sizeof(unsigned int) * edgeNum * 2);
    cudaMalloc(&t_edgeSign, sizeof(int) * edgeNum * 2);
    cudaMalloc(&t_sum, sizeof(long) * maxWarpPerGrid);
    cudaMemset(t_sum, 0, sizeof(long) * maxWarpPerGrid);

    cudaMemcpy(t_edgeOffset, d_edgeOffset, sizeof(unsigned int) * (nodeNum + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(t_edgeRow, d_edgeRow, sizeof(unsigned int) * edgeNum * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(t_edgeSign, edgeSign, sizeof(int) * edgeNum * 2, cudaMemcpyHostToDevice);

    // === Set device constants ===
    cudaMemcpyToSymbol(c_offset, &t_edgeOffset, sizeof(unsigned int *));
    cudaMemcpyToSymbol(c_row, &t_edgeRow, sizeof(unsigned int *));
    cudaMemcpyToSymbol(c_sign, &t_edgeSign, sizeof(int *));
    cudaMemcpyToSymbol(c_sums, &t_sum, sizeof(long *));
    cudaMemcpyToSymbol(c_edgeSize, &edgeNum, sizeof(unsigned int));
    cudaMemcpyToSymbol(c_nodeSize, &nodeNum, sizeof(unsigned int));
    cudaMemcpyToSymbol(c_nodeoffset, &nodeOffset, sizeof(unsigned int));
    cudaMemcpyToSymbol(c_threadsPerEdge, &curThreadsPerEdge, sizeof(int));
    cudaMemcpyToSymbol(c_nonZeroRow, &d_gpu_nonZeroRow, sizeof(int *));

    // === Allocate bitmaps ===
    unsigned int *d_bitmaps;
    cudaMalloc(&d_bitmaps, sizeof(unsigned int) * intSizePerBitmap * blockNum);
    cudaMemcpyToSymbol(c_bitmap, &d_bitmaps, sizeof(unsigned int *));

    unsigned int *d_bitmap_0, *d_bitmap_1, *d_bitmap_2;
    cudaMalloc(&d_bitmap_0, sizeof(unsigned int) * totalBlocks * nodenum_select);
    cudaMalloc(&d_bitmap_1, sizeof(unsigned int) * totalBlocks * nodenum_select);
    cudaMalloc(&d_bitmap_2, sizeof(unsigned int) * totalBlocks * nodenum_select);
    cudaMemcpyToSymbol(c_bitmap_0, &d_bitmap_0, sizeof(unsigned int *));
    cudaMemcpyToSymbol(c_bitmap_1, &d_bitmap_1, sizeof(unsigned int *));
    cudaMemcpyToSymbol(c_bitmap_2, &d_bitmap_2, sizeof(unsigned int *));

    // === Launch kernel ===
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    cudaEventRecord(start_time);
    double str_clock = clock();

    BFCbyHash2<<<blockNum, blockSize>>>(nodeNum, nodenum_select, d_gpu_nonZeroRow, gpu_nonZeroSize);

    cudaDeviceSynchronize();

    double end_clock = clock();
    cudaEventRecord(stop_time);
    cudaEventSynchronize(stop_time);

    float cudatime;
    cudaEventElapsedTime(&cudatime, start_time, stop_time);
    double elapsed_time = (end_clock - str_clock) / CLOCKS_PER_SEC;
    std::cout << "\n Exact counting kernel is done in " << elapsed_time << " secs" << std::endl;
    std::cout << " (cudaEvent) Exact counting kernel is done in " << cudatime << " ms." << std::endl;

    // === Retrieve and reduce results ===
    long *h_sum = new long[maxWarpPerGrid];
    cudaMemcpy(h_sum, t_sum, sizeof(long) * maxWarpPerGrid, cudaMemcpyDeviceToHost);
    long bfCount = thrust::reduce(h_sum, h_sum + maxWarpPerGrid);

    // === Cleanup ===
    cudaFree(t_edgeOffset);
    cudaFree(t_edgeRow);
    cudaFree(t_edgeSign);
    cudaFree(t_sum);
    cudaFree(d_gpu_nonZeroRow);
    cudaFree(d_bitmaps);
    cudaFree(d_bitmap_0);
    cudaFree(d_bitmap_1);
    cudaFree(d_bitmap_2);
    delete[] h_sum;

    cudaEventDestroy(start_time);
    cudaEventDestroy(stop_time);

    return bfCount;
}




void exact_algorithm_time_tracker() {
	double beg_clock = clock();
	cudaEvent_t start_time,stop_time;
	cudaEventCreate(&start_time);
	cudaEventCreate(&stop_time);
	cudaEventRecord(start_time);
    exact_n_bf = balanced_butterfly_counting_bucketing(cpu_vertices);
    cout << " Exact algorithm is done in " <<   exact_n_bf << " butterflies."<< endl;
	exact_n_bf = my_exact_hashmap_bfc(Edges, adjLength, edgeOffset, edgeRow, nonZeroRow);
	double end_clock = clock();
	cudaEventRecord(stop_time);
	cudaEventSynchronize(stop_time);
	float cudatime;
	cudaEventElapsedTime(&cudatime, start_time, stop_time);
	double elapsed_time1 = (end_clock - beg_clock) / CLOCKS_PER_SEC;
	//cout << " Exact algorithm is done in " << elapsed_time << "secs. There are " << exact_n_bf << " butterflies."<< endl;
    cout << "Exact algorithm is done in " << elapsed_time1 << "secs. There are "<< std::fixed << std::setprecision(0) << exact_n_bf << " butterflies." << endl;

	cout << " (cudaEvent) Exact algorithm is done in " << cudatime << " ms. " << endl;
	cudaEventDestroy(start_time);
	cudaEventDestroy(stop_time);
}



void choose_algorithm() {
	read_the_graph();
    exact_algorithm_time_tracker();
}

int main() {
	std::ios::sync_with_stdio(false);
	//int aa;

	/************GPU WARMUP***********/
	int *warmup = NULL;
	cudaMalloc(&warmup, sizeof(int));
	cudaFree(warmup);
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if(deviceCount == 0)
	{
		fprintf(stderr, "error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	int dev = 1;
	cudaSetDevice(dev);
	//int test = 0;
	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, dev) == 0)
	{
			printf("Using device %d:\n", dev);
			printf("%s; global mem: %zuB; compute v%d.%d; clock: %d kHz; shared mem: %zuB; block threads: %d; SM count: %d\n",
       devProps.name,
       devProps.totalGlobalMem,
       (int)devProps.major, (int)devProps.minor,
       (int)devProps.clockRate,
       devProps.sharedMemPerBlock,
       devProps.maxThreadsPerBlock,
       devProps.multiProcessorCount);


	}
	cout<<"GPU selected"<<endl;
	choose_algorithm();
	cerr << " Take a look at the output file ..." << endl;
	return 0;
}
