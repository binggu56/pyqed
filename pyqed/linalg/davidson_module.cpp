#include "davidson_binding.hpp"
PYBIND11_MODULE(_davidson, module) {
    pyqed::linalg::bind_davidson(module);
}
