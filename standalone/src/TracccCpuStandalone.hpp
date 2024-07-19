#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>

// Project include(s).
#include "traccc/clusterization/sparse_ccl_algorithm.hpp"
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/options/detector.hpp"

// io
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/csv/make_cell_reader.hpp"

#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Definition of the cell_order struct
struct cell_order
{
    bool operator()(const traccc::cell &lhs, const traccc::cell &rhs) const
    {
        if (lhs.module_link != rhs.module_link)
        {
            return lhs.module_link < rhs.module_link;
        }
        else if (lhs.channel1 != rhs.channel1)
        {
            return lhs.channel1 < rhs.channel1;
        }
        else
        {
            return lhs.channel0 < rhs.channel0;
        }
    }
};

// Definition of the get_module function
traccc::cell_module get_module(const std::uint64_t geometry_id,
                               const traccc::geometry *geom,
                               const traccc::digitization_config *dconfig,
                               const std::uint64_t original_geometry_id)
{
    traccc::cell_module result;
    result.surface_link = detray::geometry::barcode{geometry_id};

    // Find/set the 3D position of the detector module.
    if (geom != nullptr)
    {
        if (!geom->contains(result.surface_link.value()))
        {
            throw std::runtime_error(
                "Could not find placement for geometry ID " +
                std::to_string(result.surface_link.value()));
        }
        result.placement = (*geom)[result.surface_link.value()];
    }

    // Find/set the digitization configuration of the detector module.
    if (dconfig != nullptr)
    {
        const traccc::digitization_config::Iterator geo_it =
            dconfig->find(original_geometry_id);
        if (geo_it == dconfig->end())
        {
            throw std::runtime_error(
                "Could not find digitization config for geometry ID " +
                std::to_string(original_geometry_id));
        }

        const auto &binning_data = geo_it->segmentation.binningData();
        assert(binning_data.size() > 0);
        result.pixel.min_corner_x = binning_data[0].min;
        result.pixel.pitch_x = binning_data[0].step;
        if (binning_data.size() > 1)
        {
            result.pixel.min_corner_y = binning_data[1].min;
            result.pixel.pitch_y = binning_data[1].step;
        }
        result.pixel.dimension = geo_it->dimensions;
        result.pixel.variance_y = geo_it->variance_y;
    }

    return result;
}

std::vector<traccc::io::csv::cell> read_csv(const std::string &filename);
// std::vector<traccc::io::csv::cell> read_from_array(const std::vector<std::vector<double>> &data);
std::map<std::uint64_t, std::vector<traccc::cell>> read_deduplicated_cells(const std::vector<traccc::io::csv::cell> &cells);
std::map<std::uint64_t, std::vector<traccc::cell>> read_all_cells(const std::vector<traccc::io::csv::cell> &cells);
void read_cells(traccc::io::cell_reader_output &out, const std::vector<traccc::io::csv::cell> &cells, const traccc::geometry *geom, const traccc::digitization_config *dconfig, const std::map<std::uint64_t, detray::geometry::barcode> *barcode_map, bool deduplicate);

class TracccClusterStandalone
{
private:
    int m_deviceID;
    vecmem::host_memory_resource m_mem;
    traccc::opts::detector m_detector_opts;
    traccc::host::clusterization_algorithm m_ca;
    traccc::geometry m_surface_transforms;
    std::unique_ptr<traccc::digitization_config> m_digi_cfg;
    std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>> m_barcode_map;

public:
    TracccClusterStandalone(int deviceID = 0)
        : m_deviceID(deviceID), m_ca(m_mem)
    {
        initializePipeline();
    }

    ~TracccClusterStandalone() = default;

    void initializePipeline();
    void runPipeline(std::vector<traccc::io::csv::cell> cells);
    std::vector<traccc::io::csv::cell> read_from_array(const std::vector<std::vector<std::string>> &data);
};

void TracccClusterStandalone::initializePipeline()
{
    m_detector_opts.detector_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-detray_geometry_detray.json";
    m_detector_opts.digitization_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-digi-geometric-config.json";
    m_detector_opts.grid_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-detray_surface_grids_detray.json";
    m_detector_opts.use_detray_detector = true;

    auto geom_data = traccc::io::read_geometry(m_detector_opts.detector_file, traccc::data_format::json);
    m_surface_transforms = std::move(geom_data.first);
    m_barcode_map = std::move(geom_data.second);

    m_digi_cfg = std::make_unique<traccc::digitization_config>(traccc::io::read_digitization_config(m_detector_opts.digitization_file));
}

void TracccClusterStandalone::runPipeline(std::vector<traccc::io::csv::cell> cells)
{
    traccc::io::cell_reader_output readOut(&m_mem);

    read_cells(readOut, cells, &m_surface_transforms, m_digi_cfg.get(), m_barcode_map.get(), true);

    traccc::cell_collection_types::host data = readOut.cells;
    traccc::cell_collection_types::host &cells_per_event = readOut.cells;
    traccc::cell_module_collection_types::host &modules_per_event = readOut.modules;

    traccc::host::clusterization_algorithm::output_type measurements_per_event{&m_mem};
    measurements_per_event = m_ca(vecmem::get_data(cells_per_event), vecmem::get_data(modules_per_event));
    auto measurements_size = measurements_per_event.size();
    std::cout << "Number of measurements: " << measurements_size << std::endl;

    for (std::size_t i = 0; i < 10; ++i)
    {
        auto measurement = measurements_per_event.at(i);
        std::cout << "Measurement ID: " << measurement.measurement_id << std::endl;
        std::cout << "Local coordinates: [" << measurement.local[0] << ", " << measurement.local[1] << "]" << std::endl;
    }
}

std::vector<traccc::io::csv::cell> read_csv(const std::string &filename)
{
    std::vector<traccc::io::csv::cell> cells;
    auto reader = traccc::io::csv::make_cell_reader(filename);
    traccc::io::csv::cell iocell;

    while (reader.read(iocell))
    {
        cells.push_back(iocell);
    }

    return cells;
}

std::vector<traccc::io::csv::cell> TracccClusterStandalone::read_from_array(const std::vector<std::vector<std::string>> &data)
{
    std::vector<traccc::io::csv::cell> cells;

    for (const auto &row : data)
    {
        if (row.size() != 6)
            continue; // ensure each row contains exactly 6 elements
        traccc::io::csv::cell iocell;
        iocell.geometry_id = static_cast<std::uint64_t>(std::stoull(row[0]));
        iocell.hit_id = std::stoi(row[1]);
        iocell.channel0 = std::stoi(row[2]);
        iocell.channel1 = std::stoi(row[3]);
        iocell.timestamp = std::stoi(row[4]);
        iocell.value = std::stod(row[5]); // Assuming value is a double

        cells.push_back(iocell);
    }

    return cells;
}

std::map<std::uint64_t, std::map<traccc::cell, float, cell_order>> fill_cell_map(const std::vector<traccc::io::csv::cell> &cells, unsigned int &nduplicates)
{
    std::map<std::uint64_t, std::map<traccc::cell, float, cell_order>> cellMap;
    nduplicates = 0;

    for (const auto &iocell : cells)
    {
        traccc::cell cell{iocell.channel0, iocell.channel1, iocell.value, iocell.timestamp, 0};
        auto ret = cellMap[iocell.geometry_id].insert({cell, iocell.value});
        if (!ret.second)
        {
            cellMap[iocell.geometry_id].at(cell) += iocell.value;
            ++nduplicates;
        }
    }

    return cellMap;
}

std::map<std::uint64_t, std::vector<traccc::cell>> create_result_container(const std::map<std::uint64_t, 
                                                                            std::map<traccc::cell, float, cell_order>> &cellMap)
{
    std::map<std::uint64_t, std::vector<traccc::cell>> result;
    for (const auto &[geometry_id, cells] : cellMap)
    {
        for (const auto &[cell, value] : cells)
        {
            traccc::cell summed_cell{cell};
            summed_cell.activation = value;
            result[geometry_id].push_back(summed_cell);
        }
    }
    return result;
}

std::map<std::uint64_t, std::vector<traccc::cell>> read_deduplicated_cells(const std::vector<traccc::io::csv::cell> &cells)
{
    unsigned int nduplicates = 0;
    auto cellMap = fill_cell_map(cells, nduplicates);

    if (nduplicates > 0)
    {
        std::cout << "WARNING: " << nduplicates << " duplicate cells found." << std::endl;
    }

    return create_result_container(cellMap);
}

std::map<std::uint64_t, std::vector<traccc::cell>> read_all_cells(const std::vector<traccc::io::csv::cell> &cells)
{
    std::map<std::uint64_t, std::vector<traccc::cell>> result;

    for (const auto &iocell : cells)
    {
        traccc::cell cell{iocell.channel0, iocell.channel1, iocell.value, iocell.timestamp, 0};
        result[iocell.geometry_id].push_back(cell);
    }

    return result;
}

void read_cells(traccc::io::cell_reader_output &out, 
                const std::vector<traccc::io::csv::cell> &cells, 
                const traccc::geometry *geom, 
                const traccc::digitization_config *dconfig, 
                const std::map<std::uint64_t, 
                detray::geometry::barcode> *barcode_map, 
                bool deduplicate)
{
    auto cellsMap = (deduplicate ? read_deduplicated_cells(cells)
                                 : read_all_cells(cells));

    for (const auto &[original_geometry_id, cells] : cellsMap)
    {
        std::uint64_t geometry_id = original_geometry_id;
        if (barcode_map != nullptr)
        {
            const auto it = barcode_map->find(geometry_id);
            if (it != barcode_map->end())
            {
                geometry_id = it->second.value();
            }
            else
            {
                throw std::runtime_error(
                    "Could not find barcode for geometry ID " +
                    std::to_string(geometry_id));
            }
        }

        out.modules.push_back(
            get_module(geometry_id, geom, dconfig, original_geometry_id));
        for (auto &cell : cells)
        {
            out.cells.push_back(cell);
            out.cells.back().module_link = out.modules.size() - 1;
        }
    }
}
