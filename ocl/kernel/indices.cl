uint indices_1d_to_nd(uint index_1d) { 
	uint indices_nd[$dims];
	uint res_dim = index_1d;
	for (uint i = $dims - 1; i > 0; i--) {
		indices_nd[i] = res_dim % $strides[i];
		res_dim /= $strides[i];
	}
	indices_nd[0] = res_dim;
	return indices_nd;
 }

index_t index_nd_to_1d(
	uint indices_nd[]) {
	index_t index_1d = 0;
	for (uint i = 0; i < $dims; i++) {
		index_1d += indices_nd[i] * $strides[i];
	}
	return index_1d;
}