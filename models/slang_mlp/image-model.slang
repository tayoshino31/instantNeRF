import inline_mlp;

#define PI 3.14159265358979323846f

static const uint C = 32;
static const uint L = 5;
static const uint N_samples = 32;

//position encoding
[Differentiable]
Feature<C> embed_fn(DiffTensorView input, uint3 didx) {
    Feature<C> output;
    float[3] X;
    int idx = 0;
    for (int i = 0; i < 3; i++) {
        //float temp_in = input[didx.x , didx.y, didx.z,  i];
        float temp_in = input.loadOnce(uint4(didx.x , didx.y, didx.z,  i));
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
float relu(float x){
    return max(x, 0.0);
}

[Differentiable]
float sigmoid(float x){
    return 1.0/(1.0 + exp(-x * 0.1));
}

[Differentiable]
float[N_samples] computeAlpha(float[N_samples] sigma_a, float[N_samples] dist){
    float[N_samples] alpha;
    [ForceUnroll]
    for(int n= 0; n < N_samples; n++){
        alpha[n] = 1.0 - exp(-sigma_a[n] * dist[n]);
    }
    return alpha;
}

[Differentiable]
float[N_samples] computeComprod(float[N_samples] alpha){
    float[N_samples] comprod;
    comprod[0] = 1.0;
    [ForceUnroll]
    for(int n = 1; n < N_samples; n++){
        comprod[n] = comprod[n-1] * (1.0 - alpha[n-1] + 1e-10);
    }
    return comprod;
}

[Differentiable]
float3 volumeRendering(float[N_samples] sigma_a, 
    float[N_samples] sample_r, float[N_samples] sample_g, 
    float[N_samples] sample_b, float[N_samples] dist){  

    float[N_samples] alpha = computeAlpha(sigma_a, dist);
    float[N_samples] comprod = computeComprod(alpha);

    float3 pixel_rgb = float3(0.0,0.0,0.0);
    [ForceUnroll]
    for(int n = 0; n< N_samples; n++){
        float weight_volume = alpha[n] * comprod[n];
        pixel_rgb += weight_volume * float3(sample_r[n],sample_g[n],sample_b[n]);
    }
    return pixel_rgb;
}

[Differentiable]
[CUDAKernel]
[AutoPyBindCUDA]
void renderImage(MLP<C, 3> mlp, DiffTensorView featureGrid, DiffTensorView viewDir, 
    DiffTensorView dists, DiffTensorView imageOutput)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint row = dispatchIdx.x;
    uint col = dispatchIdx.y;

    if (row >= imageOutput.size(0) || col >= imageOutput.size(1))
        return;

    float[N_samples] sample_r;
    float[N_samples] sample_g;
    float[N_samples] sample_b;
    float[N_samples] sigma_a; 
    float[N_samples] dist;

    //store dist
    [ForceUnroll]
    for(int n = 0; n < N_samples; n++){
        //dist[n] = dists[row, col, n];
        dist[n] = dists.loadOnce(uint3(row, col, n));
    }

    //store sigma_a and sample_rgb
    for(int n = 0; n < N_samples; n++){
        uint3 didx = uint3(row, col, n);
        //get feature and view direction
        Feature<C> feature = embed_fn(featureGrid, didx);
        Feature<C> viewdir = embed_fn(viewDir, didx);
        //MLP
        mlp.Output output = mlp.eval(feature, viewdir); 
        sample_r[n] = sigmoid(output.vals[0]);
        sample_g[n] = sigmoid(output.vals[1]);
        sample_b[n] = sigmoid(output.vals[2]);
        sigma_a[n] = relu(output.vals[3]);
    }

    //volume rendering
    float3 rgb = volumeRendering(sigma_a, sample_r, sample_g, sample_b, dist);

    //store pixel color
    imageOutput.storeOnce(uint3(row, col, 0), rgb.x);
    imageOutput.storeOnce(uint3(row, col, 1), rgb.y);
    imageOutput.storeOnce(uint3(row, col, 2), rgb.z);
}

