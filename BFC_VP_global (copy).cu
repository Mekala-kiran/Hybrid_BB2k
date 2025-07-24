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
#include <thrust/sort.h>
#include "cuda_runtime.h"
#include<thrust/reduce.h>
#include<thrust/device_ptr.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/copy.h>
#include<thrust/execution_policy.h>
#include <cooperative_groups.h>

#define shareMemorySizeInBlock 128
#define hIndex 2048
#define _CRT_SECURE_NO_WARNINGS

using namespace std;
using namespace cooperative_groups;

#define SZ(x) ((int)x.size())
#define ll long long
#define ull unsigned long long
#define ld long double
#define eps 1e-11
#define bucketNum 10
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 4
#define BLOCK_BATCH 4
#define MAX_BITMAP_SIZE 8192



const int ITER_VER = 2200;
const ll shift = 1000 * 1000 * 1000LL;
const double TIME_LIMIT = 20;
const int N_WEDGE_ITERATIONS = 2 * 1000 * 1000 * 10;
const int ITERATIONS_SAMPLING = 5;
const int N_SPARSIFICATION_ITERATIONS = 5;
const int TIME_LIMIT_SPARSIFICATION = 10000; // !half an hour
const int N_FAST_EDGE_BFC_ITERATIONS = 2100; // used for fast edge sampling
const int N_FAST_WEDGE_ITERATIONS = 50; // used for fast wedge sampling

char input_address[2000], output_address [2000] ;

clock_t allStart,allEnd,tStart,tEnd;

//set < pair <int, int> > edges;
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



bool preProcess(unsigned &_nonZeroSize, unsigned int &nodeoffset,unsigned int &nodeNum_end,unsigned int &nodeNum, unsigned int &edgeNum){
    freopen(input_address, "r", stdin);
	string s;
	cin.clear();
    int size = 0;
	while (getline(cin, s)) {
 		stringstream ss; ss << s;
		vector <string> vec_str;
		for (string z; ss >> z; vec_str.push_back(z));
		if (SZ(vec_str) >= 2) {
			bool is_all_num = true;
			for (int i = 0; i < min (2, SZ(vec_str)) ; i++) is_all_num &= all_num(vec_str[i]);
			if (is_all_num) {
				int A, B, sign;
				ss.clear(); ss << vec_str[0]; ss >> A;
				ss.clear(); ss << vec_str[1]; ss >> B;
                ss.clear(); ss << vec_str[2]; ss >> sign;
				add_edge(A, B, sign);
                size++;
			}
		}
	}

	cout<<"phase : 1 finished------------   File Reading Done -------------------"<<endl;

    vertices[0].clear();
	vertices[1].clear();

	largest_index_in_partition[0] = 0;
	largest_index_in_partition[1] = SZ(vertices_in_left);
	n_vertices = SZ(vertices_in_left) + SZ(vertices_in_right);
	adj.resize(n_vertices, vector <int> ());

    edgeNum = size;
    nodeNum = n_vertices;
	nodeNum_end = SZ(vertices_in_left)>SZ(vertices_in_right)? nodeNum:SZ(vertices_in_left);
	Edges = new edge[edgeNum];
    adjLength = new int[nodeNum];
    edgeOffset = new int [nodeNum+1];
    edgeRow = new int [edgeNum*2];
    edgeSign = new int [edgeNum * 2]; // sign
    edgeOffset[0] = 0;
	nodeoffset = SZ(vertices_in_left)>SZ(vertices_in_right)? SZ(vertices_in_left): 0;


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

		//list_of_edges.push_back(make_pair(A, B));
		size++;
	}

	cout<<"----------------    phase two finished    --------------- "<<endl;

        unsigned int nodePos = 0;
        edgeOffset[0] = 0;
        for (int i = 0; i < nodeNum; i++) {
            edgeOffset[i + 1] = edgeOffset[i] + adj[i].size();
            for (auto &entry : adj[i]) {
                edgeRow[nodePos] = entry;
                edgeSign[nodePos] = edge_sign[{i, entry}]; // Correct way to retrieve sign
                nodePos++;
            }
        }


	resize_all();

	cout<<"phase three finished"<<endl;

    n_wedge_in_partition[0] = 0;
	for (int i = 0; i < largest_index_in_partition[0]; i++) {
		n_wedge_in_partition[0] += (((ll)SZ(adj[i])) * (SZ(adj[i]) - 1)) >> 1;
	}
	n_wedge_in_partition[1] = 0;
	for (int i = largest_index_in_partition[0]; i < largest_index_in_partition[1]; i++) {
		n_wedge_in_partition[1] += ((ll)SZ(adj[i]) * (SZ(adj[i]) - 1)) >> 1;
	}
	for (int i = 0; i < n_vertices; i++) {
		sort(adj[i].begin(), adj[i].end());
		sum_deg_neighbors[i] = 0;
        adjLength[i] = SZ(adj[i]);
		for (auto neighbor : adj[i]) {
			sum_deg_neighbors[i] += SZ(adj[neighbor]);
		}
	}
	cerr << " for test # edges :: " << SZ(list_of_edges) << " left :: " << SZ(vertices_in_left) << " right :: "  << SZ(vertices_in_right) << endl;
	int side = n_wedge_in_partition[0] < n_wedge_in_partition[1];
	unsigned int nonZerosize = 0;
	nonZeroRow = new int [nodeNum];
	for (int i = nodeoffset; i < nodeNum_end; i ++) {
		if (edgeOffset[i] != edgeOffset[i+1]) {
			nonZeroRow[nonZerosize++] = i;

		}
	}


/* cout << "Edge Offset:\n";
for (int i = 0; i <= nodeNum; ++i) cout << edgeOffset[i] << " ";
cout << "\nEdge Row:\n";
for (int i = 0; i < 2 * edgeNum; ++i) cout << edgeRow[i] << " ";
cout << "\nEdge Sign:\n";
for (int i = 0; i < 2 * edgeNum; ++i) cout << edgeSign[i] << " ";
cout << endl;*/

	int *tmpNonZeroRow = new int [nonZerosize];
	memcpy(tmpNonZeroRow,nonZeroRow,sizeof(int)*nonZerosize);
	delete [] nonZeroRow;
	nonZeroRow = tmpNonZeroRow;
	_nonZeroSize = nonZerosize;
	cout << "the nonZeroSize is " << nonZerosize << "node_end is "<< nodeNum_end<<endl;
	for(int j = 0; j<10; j++)
		cout<<nonZeroRow[j];
	fclose(stdin);

    return true;

}



// using namespace boost::random;
__constant__ edge *c_Edges;
__constant__ unsigned int *c_adjLen;
__constant__ unsigned int *c_offset;
__constant__ unsigned int *c_row;
__constant__ int c_threadsPerEdge;
// __constant__ long *c_sums;
__constant__ unsigned int * c_bitmap;
__device__ unsigned int *c_bitmap_0;
__device__ unsigned int *c_bitmap_1;
__device__ unsigned int *c_bitmap_2;
__device__ unsigned long long* c_sums;




__constant__ unsigned int * c_nonZeroRow;
__constant__ unsigned int c_edgeSize;
__constant__ unsigned int c_nodeSize;
__constant__ unsigned int c_nodeoffset;

//__constant__ int *c_sign;

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
	// cerr << " Insert the output file" << endl;
	// cerr << " >>> "; cin >> output_address;
	// freopen(output_address, "w", stdout);
	cerr << " ---------------------------------------------------------------------------------------------------------------------- \n";
	cerr << "| * Note that edges should be separated line by line.\n\
| In each line, the first integer number is considered as a vertex in the left partition of bipartite network, \n\
| and the second integer number is a vertex in the right partition. \n\
| In addition, multiple edges are removed from the given bipartite network.\n\
| Also, note that in this version of the source code, we did NOT remove vertices with degree zero.\n";
	cerr << " ---------------------------------------------------------------------------------------------------------------------- \n";

	cerr << " Processing the graph ... (please wait) \n";

	//get_graph();


	preProcess(nonZeroSize, nodeOffset, nodeNum_end, nodeNum, edgeNum);

	cerr << " -------------------------------------------------------------------------- \n";
	cerr << " The graph is processed - there are " << nodeNum << " vertices and " << edgeNum << " edges  \n";
	cerr << " -------------------------------------------------------------------------- \n";
}







// ---------------------- DEVICE UTILS -------------------------

__device__ int getSign(int u, int v) {
    for (unsigned int k = c_offset[u]; k < c_offset[u + 1]; ++k) {
        if (c_row[k] == v) return c_sign[k];
    }
    return -1;
}

__device__ void processTwoHop(
    int u, int v, int sign_uv,
    unsigned int* myBitmap_0,
    unsigned int* myBitmap_1,
    unsigned int* myBitmap_2
) {
    unsigned int* nbrs = c_row + c_offset[v];
    unsigned int len = c_offset[v + 1] - c_offset[v];
    for (unsigned int j = 0; j < len; ++j) {
        unsigned int w = nbrs[j];
        if (w <= u) continue;
        int sign_vw = getSign(v, w);
        int sign_sum = sign_uv + sign_vw;

        if (sign_sum == 0)
            atomicAdd(&myBitmap_0[w - c_nodeoffset], 1);
        else if (sign_sum == 1)
            atomicAdd(&myBitmap_1[w - c_nodeoffset], 1);
        else if (sign_sum == 2)
            atomicAdd(&myBitmap_2[w - c_nodeoffset], 1);
    }
}





__global__ void BFCbyHashHybrid(unsigned int totalNodeNum, unsigned int nodenum_select, int nonZeroSize) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;
    int lane_id = threadIdx.x % 32;

    unsigned int intSizePerBitmap = nodenum_select;

    // extern __shared__ unsigned int shmem[];
    // unsigned int* sh_bitmap = shmem;

    unsigned int* myBitmap_0 = c_bitmap_0 + bid * intSizePerBitmap;
    unsigned int* myBitmap_1 = c_bitmap_1 + bid * intSizePerBitmap;
    unsigned int* myBitmap_2 = c_bitmap_2 + bid * intSizePerBitmap;

    // Clear bitmaps before starting
    for (int i = tid; i < nodenum_select; i += blockDim.x) {
        myBitmap_0[i] = 0;
        myBitmap_1[i] = 0;
        myBitmap_2[i] = 0;
    }
    __syncthreads();

    for (int i = bid; i < nonZeroSize; i += gridDim.x) {
        int u = c_nonZeroRow[i];
        unsigned int* curNodeNbr = c_row + c_offset[u];
        unsigned int curNodeNbrLength = c_offset[u + 1] - c_offset[u];

        if (curNodeNbrLength <= 8) {
            if (tid == 0) {
                for (int vi = 0; vi < curNodeNbrLength; ++vi) {
                    int v = curNodeNbr[vi];
                    int sign_uv = getSign(u, v);
                    processTwoHop(u, v, sign_uv, myBitmap_0, myBitmap_1, myBitmap_2);
                }
            }
        } else if (curNodeNbrLength <= 32) {
            if (threadIdx.x < 32) {
                for (int vi = lane_id; vi < curNodeNbrLength; vi += 32) {
                    int v = curNodeNbr[vi];
                    int sign_uv = getSign(u, v);
                    processTwoHop(u, v, sign_uv, myBitmap_0, myBitmap_1, myBitmap_2);
                }
            }
        } else {
            for (int vi = tid; vi < curNodeNbrLength; vi += blockDim.x) {
                int v = curNodeNbr[vi];
                int sign_uv = getSign(u, v);
                processTwoHop(u, v, sign_uv, myBitmap_0, myBitmap_1, myBitmap_2);
            }
        }

        __syncthreads();

        // Count balanced butterflies
        unsigned long long sum = 0;
        for (int j = tid; j < nodenum_select; j += blockDim.x) {
            unsigned int two_zeros = myBitmap_0[j];
            unsigned int one_zero  = myBitmap_1[j];
            unsigned int two_ones  = myBitmap_2[j];

            unsigned int balanced = two_zeros + two_ones;
            if (balanced > 1)
                sum += ((unsigned long long)(balanced) * (balanced - 1)) / 2ULL;
            if (one_zero > 1)
                sum += ((unsigned long long)(one_zero) * (one_zero - 1)) / 2ULL;
        }

        // Warp reduce
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        if (lane_id == 0)
            atomicAdd((unsigned long long*)&c_sums[0], sum);

        __syncthreads();

        
        for (int j = tid; j < nodenum_select; j += blockDim.x) {
            myBitmap_0[j] = 0;
            myBitmap_1[j] = 0;
            myBitmap_2[j] = 0;
        }
        __syncthreads();
    }
}

ll my_exact_hashmap_bfc(edge *d_Edges, int *d_adjLen, int *d_edgeOffset, int *d_edgeRow, int *d_nonZeroRow) {
    const int blockSize = 32;
    const int blockNum = 1920;
    const int nodenum_select = nodeNum;

    // Check memory limit
    if ((blockNum * nodenum_select * sizeof(int)) / 1024 > 6 * 1024 * 1024) {
        std::cout << "RUN OUT OF GLOBAL MEMORY!!" << std::endl;
        return 0;
    }

    unsigned int *t_edgeOffset, *t_edgeRow, *t_edgeSign, *t_nonZeroRow;
    unsigned long long *t_sum;

    cudaMalloc(&t_edgeOffset, sizeof(unsigned int) * (nodeNum + 1));
    cudaMalloc(&t_edgeRow, sizeof(unsigned int) * edgeNum * 2);
    cudaMalloc(&t_edgeSign, sizeof(unsigned int) * edgeNum * 2);
    cudaMalloc(&t_nonZeroRow, sizeof(unsigned int) * nonZeroSize);
    cudaMalloc(&t_sum, sizeof(unsigned long long)); // SINGLE sum
    cudaMemset(t_sum, 0, sizeof(unsigned long long));

    cudaMemcpy(t_edgeOffset, d_edgeOffset, sizeof(unsigned int) * (nodeNum + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(t_edgeRow, d_edgeRow, sizeof(unsigned int) * edgeNum * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(t_edgeSign, edgeSign, sizeof(unsigned int) * edgeNum * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(t_nonZeroRow, d_nonZeroRow, sizeof(unsigned int) * nonZeroSize, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(c_offset, &t_edgeOffset, sizeof(unsigned int *));
    cudaMemcpyToSymbol(c_row, &t_edgeRow, sizeof(unsigned int *));
    cudaMemcpyToSymbol(c_sign, &t_edgeSign, sizeof(unsigned int *));
    cudaMemcpyToSymbol(c_nonZeroRow, &t_nonZeroRow, sizeof(unsigned int *));
    cudaMemcpyToSymbol(c_sums, &t_sum, sizeof(unsigned long long *));
    cudaMemcpyToSymbol(c_edgeSize, &edgeNum, sizeof(unsigned int));
    cudaMemcpyToSymbol(c_nodeSize, &nodeNum, sizeof(unsigned int));
    cudaMemcpyToSymbol(c_nodeoffset, &nodeOffset, sizeof(unsigned int));

    // Bitmaps
    unsigned int *d_bitmap_0, *d_bitmap_1, *d_bitmap_2;
    cudaMalloc(&d_bitmap_0, sizeof(unsigned int) * blockNum * nodenum_select);
    cudaMalloc(&d_bitmap_1, sizeof(unsigned int) * blockNum * nodenum_select);
    cudaMalloc(&d_bitmap_2, sizeof(unsigned int) * blockNum * nodenum_select);
    cudaMemcpyToSymbol(c_bitmap_0, &d_bitmap_0, sizeof(unsigned int *));
    cudaMemcpyToSymbol(c_bitmap_1, &d_bitmap_1, sizeof(unsigned int *));
    cudaMemcpyToSymbol(c_bitmap_2, &d_bitmap_2, sizeof(unsigned int *));

    // Launch
    size_t sharedMemSize = sizeof(unsigned int) * blockSize;
    BFCbyHashHybrid<<<blockNum, blockSize, sharedMemSize>>>(nodeNum, nodenum_select, nonZeroSize);
    cudaDeviceSynchronize();

    // Collect result
    unsigned long long h_sum = 0;
    cudaMemcpy(&h_sum, t_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(t_edgeOffset);
    cudaFree(t_edgeRow);
    cudaFree(t_edgeSign);
    cudaFree(t_sum);
    cudaFree(t_nonZeroRow);
    cudaFree(d_bitmap_0);
    cudaFree(d_bitmap_1);
    cudaFree(d_bitmap_2);

    return h_sum;
}





void exact_algorithm_time_tracker() {
	double beg_clock = clock();
	cudaEvent_t start_time,stop_time;
	cudaEventCreate(&start_time);
	cudaEventCreate(&stop_time);
	cudaEventRecord(start_time);
	exact_n_bf = my_exact_hashmap_bfc(Edges, adjLength, edgeOffset, edgeRow, nonZeroRow);
	double end_clock = clock();
	cudaEventRecord(stop_time);
	cudaEventSynchronize(stop_time);
	float cudatime;
	cudaEventElapsedTime(&cudatime, start_time, stop_time);
	double elapsed_time = (end_clock - beg_clock) / CLOCKS_PER_SEC;
	//cout << " Exact algorithm is done in " << elapsed_time << "secs. There are " << exact_n_bf << " butterflies."<< endl;
cout << "Exact algorithm is done in " << elapsed_time << "secs. There are "
     << std::fixed << std::setprecision(0) << exact_n_bf << " butterflies." << endl;

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
	int aa;

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
	int test = 0;
	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, dev) == 0)
	{
			printf("Using device %d:\n", dev);
			printf("%s; global mem: %luB; compute v%d.%d; clock: %d kHz; shared mem: %dB; block threads: %d; SM count: %d\n",
							devProps.name, devProps.totalGlobalMem,
							(int)devProps.major, (int)devProps.minor,
							(int)devProps.clockRate,
							devProps.sharedMemPerBlock, devProps.maxThreadsPerBlock, devProps.multiProcessorCount);
	}
	cout<<"GPU selected"<<endl;
	choose_algorithm();
	cerr << " Take a look at the output file ..." << endl;
	return 0;
}
