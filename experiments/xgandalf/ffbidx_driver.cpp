// ffbidx driver: GPU fast-feedback indexer (known-cell) on the SAME rlp frames.
// Reads frames.txt (FRAME id npk + npk "x y z" reciprocal spots), uses the lysozyme
// cell as the prior B0, prints "id v0x v0y v0z v1x v1y v1z v2x v2y v2z score ms"
// (best output cell) or "id NONE ms".
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include "ffbidx/c_api.h"

int main(int argc, char** argv) {
    std::string path = argv[1];
    std::ifstream in(path);
    // lysozyme real-space cell vectors (Angstrom), tetragonal aligned to axes
    float ax[3] = {79.02f, 0.f, 0.f};
    float bx[3] = {0.f, 79.02f, 0.f};
    float cx[3] = {0.f, 0.f, 37.98f};

    config_persistent cpers; config_runtime crt; config_ifssr cifssr;
    set_defaults(&cpers, &crt, &cifssr);
    cpers.max_spots = 700; cpers.max_input_cells = 1;
    if (cpers.max_output_cells < 32) cpers.max_output_cells = 32;

    std::vector<char> emsg(256); error err = { emsg.data(), (unsigned)emsg.size() };
    if (check_config(&cpers, &crt, &cifssr, &err) == -1) { fprintf(stderr,"cfg: %s\n",emsg.data()); return 2; }
    std::vector<float> X(cpers.max_spots+3), Y(cpers.max_spots+3), Z(cpers.max_spots+3);
    std::vector<float> buf((3*3+1)*cpers.max_output_cells);
    output out = { &buf[0], &buf[3*cpers.max_output_cells], &buf[6*cpers.max_output_cells],
                   &buf[9*cpers.max_output_cells], cpers.max_output_cells };

    std::string tok;
    while (in >> tok) {
        if (tok != "FRAME") break;
        int fid, npk; in >> fid >> npk;
        // cell vectors at index 0..2
        X[0]=ax[0]; Y[0]=ax[1]; Z[0]=ax[2];
        X[1]=bx[0]; Y[1]=bx[1]; Z[1]=bx[2];
        X[2]=cx[0]; Y[2]=cx[1]; Z[2]=cx[2];
        int ns = npk; if (ns > (int)cpers.max_spots) ns = cpers.max_spots;
        for (int i=0;i<npk;i++){ float a,b,c; in>>a>>b>>c; if(i<ns){X[3+i]=a;Y[3+i]=b;Z[3+i]=c;} }
        input inp = { {&X[0],&Y[0],&Z[0]}, {&X[3],&Y[3],&Z[3]}, 1u, (unsigned)ns, true, true };
        auto t0 = std::chrono::high_resolution_clock::now();
        int h = create_indexer(&cpers, &err, nullptr);
        if (h == -1) { fprintf(stderr,"create: %s\n",emsg.data()); return 2; }
        int idx = indexer_op(h, &inp, &out, &crt, &cifssr);
        drop_indexer(h);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        if (idx < 0) { printf("%d NONE %.3f\n", fid, ms); }
        else {
            printf("%d", fid);
            for (int j=0;j<3;j++) printf(" %.5f %.5f %.5f", out.x[3*idx+j], out.y[3*idx+j], out.z[3*idx+j]);
            printf(" %.5f %.3f\n", out.score[idx], ms);
        }
        fflush(stdout);
    }

    return 0;
}
