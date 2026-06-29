// xgandalf driver: index the SAME rlp q-vectors fftindex used, via the production
// libxgandalf (CrystFEL 0.12.0). Reads frames.txt (FRAME id npk + npk "x y z" lines),
// runs IndexerPlain per frame, prints "id  b00..b22  ms" (or "id NONE ms").
//
//   mode=blind : ExperimentSettings(min/max real-vec-length) -- no cell prior
//   mode=known : ExperimentSettings(sampleReciprocalLattice, tolerance) -- lysozyme cell
//
// xgandalf settings = CrystFEL defaults (sampling-pitch 6 = denseWithSecondaryMiller,
// grad-desc 4 = manyMany). Output basis is converted to a real cell + scored with
// fftindex's own same_lattice in compare.py, so the accuracy metric is identical.
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <xgandalf/ExperimentSettings.h>
#include <xgandalf/IndexerPlain.h>
#include <xgandalf/Lattice.h>

using namespace xgandalf;
using namespace Eigen;

int main(int argc, char** argv)
{
    std::string path = argv[1];
    std::string mode = argc > 2 ? argv[2] : "blind";
    std::ifstream in(path);

    // detector/beam settings (used by xgandalf for resolution/tolerance sizing);
    // reflectionRadius ~ fftindex's inlier tol (0.02*qmax, qmax~0.5 -> 0.01 1/A).
    float beamE = 9500, detDist = 0.246, detR = 0.045, diverg = 0.05, nonMono = 0.0025, reflR = 0.01;
    ExperimentSettings* es;
    if (mode == "known") {
        Matrix3f recip = Matrix3f::Zero();        // columns a*,b*,c* of tetragonal lysozyme
        recip(0, 0) = 1.0f / 79.02f;
        recip(1, 1) = 1.0f / 79.02f;
        recip(2, 2) = 1.0f / 37.98f;
        Lattice lat(recip);
        es = new ExperimentSettings(beamE, detDist, detR, diverg, nonMono, lat, 0.05f, reflR);
    } else {
        es = new ExperimentSettings(beamE, detDist, detR, diverg, nonMono, 25.0f, 100.0f, reflR);
    }
    IndexerPlain indexer(*es);
    indexer.setSamplingPitch(IndexerPlain::SamplingPitch::denseWithSeondaryMillerIndices);
    indexer.setGradientDescentIterationsCount(IndexerPlain::GradientDescentIterationsCount::manyMany);

    std::string tok;
    while (in >> tok) {
        if (tok != "FRAME") break;
        int fid, npk;
        in >> fid >> npk;
        Matrix3Xf peaks(3, npk);
        for (int i = 0; i < npk; i++)
            in >> peaks(0, i) >> peaks(1, i) >> peaks(2, i);

        std::vector<Lattice> lattices;
        auto t0 = std::chrono::high_resolution_clock::now();
        indexer.index(lattices, peaks);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (lattices.empty()) {
            std::cout << fid << " NONE " << ms << "\n";
        } else {
            Matrix3f b = lattices[0].getBasis();
            std::cout << fid;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    std::cout << " " << b(r, c);
            std::cout << " " << ms << "\n";
        }
    }
    return 0;
}
