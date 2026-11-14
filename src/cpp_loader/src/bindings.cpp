
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "binpack_loader.h"
namespace py = pybind11; using namespace binpack;
PYBIND11_MODULE(cpp_loader, m) {
    py::class_<BinpackDataset>(m, "BinpackDataset")
        .def(py::init<const std::string&>())
        .def("__len__", &BinpackDataset::size)
        .def("__getitem__", [](const BinpackDataset& self, size_t idx) {
            auto pos = self[idx];
            return py::make_tuple(
                py::array_t<uint16_t>(pos.features),
                py::array_t<float>(pos.weights),
                py::array_t<float>(std::vector<float>(pos.wdl.begin(), pos.wdl.end())),
                pos.eval, pos.game_result
            );
        });
    m.doc() = "Fast C++ binpack loader";
}
