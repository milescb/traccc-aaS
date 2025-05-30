#include "TracccGpuStandalone.hpp"

void TracccGpuStandalone::initialize()
{
    // HACK: hard code location of detector and digitization file
    m_detector_opts.detector_file = "/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-00/ITk_DetectorBuilder_geometry.json";
    m_detector_opts.digitization_file = "/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-00/ITk_digitization_config_with_strips.json";
    m_detector_opts.grid_file = "/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-00/ITk_DetectorBuilder_surface_grids.json";
    m_detector_opts.material_file = "";

    // Read the detector description
    traccc::io::read_detector_description(
        m_det_descr, m_detector_opts.detector_file,
        m_detector_opts.digitization_file, traccc::data_format::json);
    traccc::silicon_detector_description::data m_det_descr_data{
        vecmem::get_data(m_det_descr)};
    m_device_det_descr = traccc::silicon_detector_description::buffer(
            static_cast<traccc::silicon_detector_description::buffer::size_type>(
                m_det_descr.size()),
            *m_device_mr);
    m_copy.setup(m_device_det_descr)->wait();
    m_copy(m_det_descr_data, m_device_det_descr)->wait();

    // Create the detector and read the configuration file
    m_detector = std::make_unique<host_detector_type>(*m_host_mr);
    traccc::io::read_detector(
        *m_detector, *m_host_mr, m_detector_opts.detector_file,
        m_detector_opts.material_file, m_detector_opts.grid_file);
    
    // copy it to the device - dereference the unique_ptr to get the actual object
    m_device_detector = detray::get_buffer(*m_detector, *m_device_mr, m_copy);
    m_stream.synchronize();
    m_device_detector_view = detray::get_data(m_device_detector);

    return;
}

traccc::track_state_container_types::host TracccGpuStandalone::run(
    traccc::edm::spacepoint_collection::host spacepoints_per_event,
    traccc::measurement_collection_types::host measurements_per_event)
{   
    // copy spacepoints and measurements to device
    traccc::edm::spacepoint_collection::buffer spacepoints(
        static_cast<unsigned int>(spacepoints_per_event.size()), *m_cached_device_mr);
    m_copy(vecmem::get_data(spacepoints_per_event), spacepoints)->wait();

    traccc::measurement_collection_types::buffer measurements(
        static_cast<unsigned int>(measurements_per_event.size()), *m_cached_device_mr);
    m_copy(vecmem::get_data(measurements_per_event), measurements)->wait();

    // Seeding and track param est.
    auto seeds = m_seeding(spacepoints);
    m_stream.synchronize();

    const traccc::cuda::track_params_estimation::output_type track_params =
        m_track_parameter_estimation(measurements, spacepoints,
            seeds, m_field_vec);
    m_stream.synchronize();

    // track finding                        
    // Run the track finding
    const finding_algorithm::output_type track_candidates = m_finding(
        m_device_detector_view, m_field, measurements, track_params);

    // Run the track fitting
    const fitting_algorithm::output_type track_states = 
        m_fitting(m_device_detector_view, m_field, track_candidates);

    // Print fitting stats
    std::cout << "Number of measurements: " << measurements_per_event.size() << std::endl;
    std::cout << "Number of spacepoints: " << spacepoints_per_event.size() << std::endl;
    std::cout << "Number of seeds: " << m_copy.get_size(seeds) << std::endl;
    std::cout << "Number of track params: " << m_copy.get_size(track_params) << std::endl;
    std::cout << "Number of track candidates: " << track_candidates.items.size() << std::endl;
    std::cout << "Number of fitted tracks: " << track_states.items.size() << std::endl;

    // copy track states to host
    //! Expensive with so many gd track states
    auto track_states_host = m_copy_track_states(track_states);

    // run ambiguity resolution
    // TODO: this should run before fitting, but requires copy back and forth first
    // TODO: ask traccc people about this
    // traccc::track_state_container_types::host resolved_track_states_cuda =
    //     m_resolution_alg(traccc::get_data(track_states_host));

    return track_states_host;
}

void TracccGpuStandalone::read_spacepoints(
    traccc::edm::spacepoint_collection::host& spacepoints,
    std::vector<clusterInfo>& detray_clusters, bool do_strip)
{
    traccc::measurement_collection_types::host measurements;
    read_measurements(measurements, detray_clusters, do_strip);

    std::map<traccc::geometry_id, unsigned int> m;
    for(std::vector<clusterInfo>::size_type i = 0; i < detray_clusters.size();i++){
        clusterInfo cluster = detray_clusters[i];
        if(do_strip && cluster.pixel){continue;}

        // Construct the local 3D(2D) position of the measurement.
        // traccc::measurement meas;
        // meas = measurements[i];

        spacepoints.push_back({static_cast<unsigned int>(i), 
            traccc::edm::spacepoint_collection::host::INVALID_MEASUREMENT_INDEX,
            {static_cast<float>(cluster.globalPosition[0]),
            static_cast<float>(cluster.globalPosition[1]),
            static_cast<float>(cluster.globalPosition[2])},
            0.f, 0.f});
    }
}


void TracccGpuStandalone::read_measurements(
    traccc::measurement_collection_types::host& measurements,
    std::vector<clusterInfo>& detray_clusters, bool do_strip)
{
    std::map<traccc::geometry_id, unsigned int> m;
    std::multimap<uint64_t,detray::geometry::barcode> sf_seen;

    for(std::vector<clusterInfo>::size_type i = 0; i < detray_clusters.size();i++){

        clusterInfo cluster = detray_clusters[i];
        if(do_strip && cluster.pixel){continue;}

        uint64_t geometry_id = cluster.detray_id;
        const auto& sf = detray::geometry::barcode{geometry_id};
        const detray::tracking_surface surface{*m_detector, sf};
        cluster.localPosition[0] = cluster.localPosition.x();
        cluster.localPosition[1] = cluster.localPosition.y();

        // ATH_MSG_INFO("Traccc measurement at index " << i << ": " << cluster.localPosition[0] << "," << cluster.localPosition[1]);

        // Construct the measurement object.
        traccc::measurement meas;
        std::array<detray::dsize_type<traccc::default_algebra>, 2u> indices{0u, 0u};
        meas.meas_dim = 0u;
        for (unsigned int ipar = 0; ipar < 2u; ++ipar) {
            if (((cluster.local_key) & (1 << (ipar + 1))) != 0) {

                switch (ipar) {
                    case 0: {
                        meas.local[0] = cluster.localPosition.x();
                        meas.variance[0] = 0.0025;
                        indices[meas.meas_dim++] = ipar;
                    }; break;
                    case 1: {
                        meas.local[1] = cluster.localPosition.y();
                        meas.variance[1] = 0.0025;
                        indices[meas.meas_dim++] = ipar;
                    }; break;
                }
            }
        }

        meas.subs.set_indices(indices);
        meas.surface_link = detray::geometry::barcode{geometry_id};

        // Keeps measurement_id for ambiguity resolution
        meas.measurement_id = i;
        measurements.push_back(meas);
    }
}


std::vector<clusterInfo> TracccGpuStandalone::read_clusters_from_csv(
    const std::string& filename)
{
    std::vector<clusterInfo> clusters;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    // Read and discard the header line
    if (!std::getline(file, line)) {
         throw std::runtime_error("Could not read header line from file: " + filename);
    }

    // Read data lines
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        clusterInfo cluster;
        int col_index = 0;
        double global_x, global_y, global_z;
        double local_x, local_y;
        int pixel_val;

        try {
            // Read each field separated by a comma
            while (std::getline(ss, field, ',')) {
                // Trim leading/trailing whitespace
                field.erase(0, field.find_first_not_of(" \t\n\r\f\v"));
                field.erase(field.find_last_not_of(" \t\n\r\f\v") + 1);

                switch (col_index) {
                    case 0: // atlas_id (ignored)
                        break;
                    case 1: // detray_id
                        cluster.detray_id = std::stoull(field);
                        break;
                    case 2: // measurement_id (ignored)
                        break;
                    case 3: // local_key
                        cluster.local_key = std::stoul(field);
                        break;
                    case 4: // local_x
                        local_x = std::stod(field);
                        break;
                    case 5: // local_y
                        local_y = std::stod(field);
                        break;
                    case 6: // global_x
                        global_x = std::stod(field);
                        break;
                    case 7: // global_y
                        global_y = std::stod(field);
                        break;
                    case 8: // global_z
                        global_z = std::stod(field);
                        break;
                    case 9: // pixel
                        pixel_val = std::stoi(field);
                        cluster.pixel = (pixel_val != 0);
                        break;
                    default:
                        // Handle unexpected extra columns if necessary
                        break;
                }
                col_index++;
            }
             // Check if we read the expected number of columns
            if (col_index != 10) {
                 std::cerr << "Warning: Row has " << col_index << " columns, expected 10. Line: " << line << std::endl;
                 continue; // Skip this row or handle error as appropriate
            }

            // Assign Eigen vectors after reading all components
            cluster.globalPosition = Eigen::Vector3d(global_x, global_y, global_z);
            cluster.localPosition = Eigen::Vector2d(local_x, local_y);

            clusters.push_back(cluster);

        } catch (const std::invalid_argument& e) {
            std::cerr << "Warning: Invalid argument during conversion for line: " << line << ". Error: " << e.what() << std::endl;
            // Skip this row or handle error as appropriate
            continue;
        } catch (const std::out_of_range& e) {
            std::cerr << "Warning: Out of range during conversion for line: " << line << ". Error: " << e.what() << std::endl;
            // Skip this row or handle error as appropriate
            continue;
        }
    }

    file.close();

    // print first 5 clusters to ensure they are read correctly
    std::cout << "Read " << clusters.size() << " clusters from file." << std::endl;
    for (size_t i = 0; i < std::min(clusters.size(), size_t(5)); ++i) {
        std::cout << "Cluster " << i << ": "
                  << "detray_id: " << clusters[i].detray_id
                  << ", local_key: " << clusters[i].local_key
                  << ", globalPosition: (" << clusters[i].globalPosition.transpose() << ")"
                  << ", localPosition: (" << clusters[i].localPosition.transpose() << ")"
                  << ", pixel: " << clusters[i].pixel
                  << std::endl;
    }

    return clusters;
}

std::vector<clusterInfo> TracccGpuStandalone::read_from_array(
    const std::vector<std::uint64_t> &geometry_ids,
    const std::vector<std::vector<double>> &data)
{
    std::vector<clusterInfo> clusters;

    if (geometry_ids.size() != data.size())
    {
        throw std::runtime_error("Number of geometry IDs and data rows do not match.");
    }

    for (size_t i = 0; i < data.size(); ++i) 
    {
        const auto& row = data[i];
        if (row.size() != 7)
            continue; 

        clusterInfo cluster;

        if (i < geometry_ids.size()) 
        {
            cluster.detray_id = geometry_ids[i];
        } 
        else 
        {
            continue;
        }

        cluster.local_key = static_cast<unsigned int>(row[0]);
        cluster.localPosition = Eigen::Vector2d(row[1], row[2]);
        cluster.globalPosition = Eigen::Vector3d(row[3], row[4], row[5]);
        cluster.pixel = (row[6] != 0);
        clusters.push_back(cluster);
    }

    return clusters;
}


int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Not enough arguments, minimum requirement two of the form: " << std::endl;
        std::cout << argv[0] << " <event_file> " << "<deviceID>" << std::endl;
        return -1;
    }

    std::string event_file = std::string(argv[1]);
    int deviceID = std::stoi(argv[2]);

    std::cout << "Using device ID: " << deviceID << std::endl;
    std::cout << "Running " << argv[0] << " on " << event_file << std::endl;

    vecmem::host_memory_resource host_mr;
    vecmem::cuda::device_memory_resource device_mr(deviceID);
    
    TracccGpuStandalone traccc_gpu(&host_mr, &device_mr, deviceID);
   
    traccc::edm::spacepoint_collection::host spacepoints(host_mr);
    traccc::measurement_collection_types::host measurements(&host_mr);

    std::vector<clusterInfo> detray_clusters = traccc_gpu.read_clusters_from_csv(event_file);
    traccc_gpu.read_measurements(measurements, detray_clusters, false);
    traccc_gpu.read_spacepoints(spacepoints, detray_clusters, false);

    std::cout << "Number of measurements: " << measurements.size() << std::endl;
    std::cout << "Number of spacepoints: " << spacepoints.size() << std::endl;

    // run the traccc algorithm
    auto traccc_result = traccc_gpu.run(spacepoints, measurements);

    return 0;
}