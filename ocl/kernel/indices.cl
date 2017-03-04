#define DECLARE_INDICES_FUNCS(tid) \
uint[] indices_1d_to_nd_##tid(uint index_1d) { \ 
	uint indices_nd[dims##tid]; \
	uint res_dim = index_1d; \
	for (uint i = dims##tid - 1; i > 0; i--) {\
		indices_nd[i] = res_dim % strides##tid[i];\
		res_dim /= strides##tid[i];\
	}\
	indices_nd[0] = res_dim;\
	return indices_nd;\
 }\
uint index_nd_to_1d_##tid(uint indices_nd[]) {\
	index_t index_1d = 0;\
	for (uint i = 0; i < dims##tid; i++) {\
		index_1d += indices_nd[i] * strides##tid[i];\
	}\
	return index_1d;\
}\