import inline_mlp;

#define PI 3.14159265358979323846f

static const uint C = 32;
static const uint L = 5;

[Differentiable]
Feature<C> embed_fn(DiffTensorView input, uint3 didx) {
    Feature<C> output;
    float[3] X;
    int idx = 0;

    for (int i = 0; i < 3; i++) {
        float temp_in = input[didx.x , didx.y, didx.z,  i];
        output.vals[idx++] = temp_in;
        X[i] = temp_in;
    }

    bool done = false;
    for(int i = 0; i < L; i++){ 
        for(int k = 0; k < 3; k ++){
            if(idx == C){
                done = true;
                break;
            }
            output.vals[idx] = sin(pow(2.0, i)* X[k]);
            idx ++;
        }
        for(int k = 0; k < 3; k ++){
            if(idx == C){
                done = true;
                break;
            }
            output.vals[idx] = cos(pow(2.0, i)* X[k]);
            idx ++;
        }
        if(done) break;
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
    uint3 didx = uint3(dispatchIdx.x, dispatchIdx.y,dispatchIdx.z );
    Feature<C> feature = embed_fn(featureGrid, didx);
    Feature<C> viewdir = embed_fn(viewDir, didx);

    mlp.Output output = mlp.eval(feature, viewdir); 

    for (int i = 0; i < 4; i++) {
        imageOutput.storeOnce(uint4(dispatchIdx.x, dispatchIdx.y, dispatchIdx.z, i), output.vals[i]);
    }
}