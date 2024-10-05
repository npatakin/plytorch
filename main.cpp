#include <cstring>
#include <string>
#include <stdexcept>

#include <fstream>
#include <iostream>

#include <torch/extension.h>
#include <pybind11/eval.h>

#include "miniply.h"


using namespace miniply;
using PropertiesType = std::unordered_map<std::string, torch::Tensor>;
using ElementsType = std::unordered_map<std::string, PropertiesType>;


const std::unordered_map<PLYPropertyType, torch::ScalarType> ply_to_torch_dtype = {
    {PLYPropertyType::Char, torch::kByte},
    {PLYPropertyType::UChar, torch::kByte},
    {PLYPropertyType::Short, torch::kInt16},
    {PLYPropertyType::UShort, torch::kUInt16},
    {PLYPropertyType::Int, torch::kInt32},
    {PLYPropertyType::UInt, torch::kUInt32},
    {PLYPropertyType::Float, torch::kFloat},
    {PLYPropertyType::Double, torch::kDouble},
};

const std::unordered_map<torch::ScalarType, std::string> torch_dtype_to_ply = {
        {torch::kUInt8, "uchar"},
        {torch::kInt8, "char"},
        {torch::kUInt16, "ushort"},
        {torch::kInt16, "short"},
        {torch::kUInt32, "uint"},
        {torch::kInt32, "int"},
        {torch::kFloat32, "float"},
        {torch::kFloat64, "double"}
};

const std::unordered_map<torch::ScalarType, uint32_t> torch_dtype_to_size = {
        {torch::kUInt8, 1},
        {torch::kInt8, 1},
        {torch::kUInt16, 2},
        {torch::kInt16, 2},
        {torch::kUInt32, 4},
        {torch::kInt32, 4},
        {torch::kFloat32, 4},
        {torch::kFloat64, 8}
};


torch::ScalarType get_torch_dtype(PLYPropertyType type) {
    auto result = ply_to_torch_dtype.find(type);
    if (result != ply_to_torch_dtype.end()) {
        return result->second;
    } else {
        throw std::runtime_error("unknown PLYProperty dtype");
    }
}

std::string get_ply_dtype(torch::ScalarType t) {
    auto result = torch_dtype_to_ply.find(t);
    if (result != torch_dtype_to_ply.end()) {
        return result->second;
    } else {
        throw std::runtime_error("cannot convert torch dtype '" + std::string(toString(t)) + "' into PLY type");
    }
}

uint32_t get_torch_dtype_size(torch::ScalarType t) {
    auto result = torch_dtype_to_size.find(t);
    if (result != torch_dtype_to_size.end()) {
        return result->second;
    } else {
        throw std::runtime_error("cannot convert torch dtype '" + std::string(toString(t)) + "' into PLY type");
    }
}

std::pair<std::string, PropertiesType> read_ply_element(miniply::PLYReader& reader, int element_idx) {
    auto element = reader.get_element(element_idx);
    PropertiesType props_dict;
    uint32_t N = element->count;
    std::vector<std::string> prop_names;

    uint32_t i = 0;
    for (const auto & property : element->properties) {
        std::string prop_name = property.name;
        prop_names.push_back(prop_name);

        if (property.countType != PLYPropertyType::None) {
            torch::ScalarType prop_dtype = get_torch_dtype(property.type);

            std::vector<uint32_t> rowcounts;
            std::copy(property.rowCount.begin(), property.rowCount.end(), std::back_inserter(rowcounts));
            std::sort(rowcounts.begin(), rowcounts.end());
            auto last = std::unique(rowcounts.begin(), rowcounts.end());
            rowcounts.erase(last, rowcounts.end());
            if (rowcounts.size() != 1) {
                throw std::runtime_error("list property '" + property.name + "' has varying rowcount(from "+std::to_string(rowcounts[0])+" to "+std::to_string(rowcounts[rowcounts.size() - 1])+"), which is not supported!");
            }

            props_dict[prop_name] = torch::empty({N, rowcounts[0]},
                                                 at::TensorOptions().dtype(prop_dtype).device(torch::kCPU));
            reader.extract_list_property(i, property.type, props_dict[prop_name].data_ptr());
        } else {
            torch::ScalarType prop_dtype = get_torch_dtype(property.type);
            props_dict[prop_name] = torch::empty({N,},
                                                 at::TensorOptions().dtype(prop_dtype).device(torch::kCPU));

            reader.extract_properties(&i, 1, property.type, props_dict[prop_name].data_ptr());
        }
        ++i;
    }

    return {element->name, props_dict};
}

ElementsType read_ply(const std::string& path) {
    miniply::PLYReader reader(path.c_str());

    if (!reader.valid()) {
        throw std::runtime_error("Failed to open specified path: " + path);
    }

    ElementsType result;

    for (int i = 0; i != reader.num_elements(); ++i) {
        reader.load_element();
        auto [el_name, element] = read_ply_element(reader, i);
        result[el_name] = element;
        reader.next_element();
    }
    return result;
}


void pyprint(const std::string& msg) {
    py::exec("print('"+msg+"')");
}

bool is_big_endian(void)
{
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};

    return bint.c[0] == 1;
}


bool write_ply(const std::string& path, const ElementsType& elements) {
    std::ofstream mesh_file(path, std::ios::binary | std::ios::out);

    mesh_file << "ply\n";
    if (is_big_endian()) {
        mesh_file << "format binary_big_endian 1.0\n";
    } else {
        mesh_file << "format binary_little_endian 1.0\n";
    }

    for (const auto &[element_name, element]: elements) {
        mesh_file << "element " << element_name << " " << element.begin()->second.size(0) << "\n";
        for (const auto& [property_name, data]: element) {
            if ((data.ndimension() > 1) && (data.size(1) > 1)) {
                mesh_file << "property list uchar ";
            } else {
                mesh_file << "property ";
            }
            mesh_file << get_ply_dtype(data.scalar_type()) << " " << property_name << "\n";
        }
    }
    mesh_file << "end_header\n";

    for (const auto &[element_name, element]: elements) {
        std::vector<char *> src_ptrs(element.size());
        std::vector<uint32_t> strides(element.size());
        std::vector<char> is_list(element.size());
        std::vector<char> list_sizes(element.size(), 0);

        uint32_t num_rows = element.begin()->second.size(0);

        int i = 0;
        for (const auto &[property_name, data]: element) {
            src_ptrs[i] = (char *) data.data_ptr();
            strides[i] = get_torch_dtype_size(data.scalar_type());
            is_list[i] = char((data.ndimension() > 1) && (data.size(1) > 1));
            if (is_list[i]) {
                list_sizes[i] = data.size(1);
            }
            ++i;
        }

        for (uint32_t el_idx = 0; el_idx != num_rows; ++el_idx) {
            for (uint32_t prop_idx = 0; prop_idx != element.size(); ++prop_idx) {
                uint32_t cur_stride = strides[prop_idx];
                if (is_list[prop_idx] == 0) {
                    mesh_file.write(src_ptrs[prop_idx], cur_stride);
                    src_ptrs[prop_idx] += cur_stride;
                } else {
                    mesh_file.write(&list_sizes[prop_idx], 1);
                    mesh_file.write(src_ptrs[prop_idx], cur_stride * list_sizes[prop_idx]);
                    src_ptrs[prop_idx] += cur_stride * list_sizes[prop_idx];
                }
            }
        }
    }

    mesh_file.close();
    return true;
}


std::pair<torch::Tensor, std::vector<std::string>> read_float_ply(const std::string& path) {
    miniply::PLYReader reader(path.c_str());
    if (!reader.valid()) {
        throw std::runtime_error("Failed to open specified path: " + path);
    }

    if (reader.num_elements() != 1) {
        throw std::runtime_error("Invalid PLY file");
    }
    const miniply::PLYElement *elem = reader.get_element(0);

    if (reader.file_type() != miniply::PLYFileType::Binary) {
        throw std::runtime_error("Unsupported PLY file encoding");
    }

    std::vector<std::string> props(elem->properties.size());
    for (int i = 0; i != props.size(); ++i) {
        props[i] = elem->properties[i].name;
    }

    std::ifstream fs(path, std::ios::binary);
    torch::Tensor data = torch::empty({elem->count, static_cast<long>(elem->properties.size())},
                                      at::TensorOptions().dtype<float>().device(torch::kCPU));

    size_t data_size = size_t(elem->count) * elem->properties.size() * 4;
    fs.seekg(-data_size, std::ios_base::end);
    fs.read(static_cast<char *>(data.data_ptr()), data_size);
    fs.close();

    return {data, props};
}

void write_float_ply(
        const std::string& path,
        const torch::Tensor& data,
        const std::vector<std::string>& props
)
{
    std::ofstream of(path, std::ios::binary);
    if (!of.is_open()) {
        throw std::runtime_error("Could not create file: " + path);
    }
    if (data.ndimension() != 2) {
        throw std::runtime_error("Data tensor must be 2-dimensional");
    }
    if (data.size(1) != props.size()) {
        throw std::runtime_error("tensor.size(1) != len(props)");
    }
    if (data.device() != torch::kCPU) {
        throw std::runtime_error("data must be on CPU");
    }
    if (data.dtype() != torch::kFloat32) {
        throw std::runtime_error("data dtype must be float32");
    }

    of << "ply" << std::endl <<
          "format binary_little_endian 1.0"<< std::endl <<
          "element vertex " << data.size(0) << std::endl;

    for (const auto& prop: props) {
        of << "property float " << prop << std::endl;
    }
    of << "end_header" << std::endl;

    size_t data_size = 4 * data.size(0) * data.size(1);
    of.write(static_cast<const char *>(data.data_ptr()), data_size);
    of.close();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read_float_ply", &read_float_ply, "Read gaussian point cloud PLY file");
    m.def("write_float_ply", &write_float_ply, "Write gaussian point cloud PLY file");
    m.def("read_ply", &read_ply, "Read generic PLY file");
    m.def("write_ply", &write_ply, "Write generic PLY file");
}