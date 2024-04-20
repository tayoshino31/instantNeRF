import inline_mlp;

#define PI 3.14159265358979323846f

static const uint C = 32;

[Differentiable]
Feature<C> getInFeature(DiffTensorView input, uint3 idx) {
    Feature<C> output;
    for (int i = 0; i < C; i++) {
        output.vals[i] = input[idx.x , idx.y, idx.z,  i];
    }
    return output;
}

[Differentiable]
[CUDAKernel]
[AutoPyBindCUDA]
void renderImage(MLP<C, 3> mlp, DiffTensorView featureGrid, DiffTensorView viewDir, DiffTensorView imageOutput)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= imageOutput.size(0) || dispatchIdx.y >= imageOutput.size(1)|| dispatchIdx.z >= imageOutput.size(2))
        return;

    // Compute the input feature for our current pixel.
    uint3 idx = uint3(dispatchIdx.x, dispatchIdx.y,dispatchIdx.z );
    Feature<C> feature = getInFeature(featureGrid, idx);
    Feature<C> viewdir = getInFeature(viewDir, idx);

    mlp.Output output = mlp.eval(feature, viewdir); 

    for (int i = 0; i < 4; i++) {
        imageOutput.storeOnce(uint4(dispatchIdx.x, dispatchIdx.y, dispatchIdx.z, i), output.vals[i]);
    }
}