// updating the module: conda env update -f environment.yml && conda activate gp && rm -rf build/ && pip install .

#include <iostream>
#include <cmath>
#include <vector>
#include <time.h>
#include <chrono>
#include <functional>
#include <string>
#include <map>

#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/chrono.h>

#include <Eigen/Core>
#include <Eigen/Dense>

typedef Eigen::Vector2d Vector2d;
typedef Eigen::VectorXd Vector;
typedef Eigen::Ref<Vector> RefVector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Ref<Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> RefMatrix;
typedef Eigen::VectorXi VectorXi;
typedef Eigen::Ref<VectorXi> RefVectorXi;

typedef std::vector<std::vector<std::vector<int> > > vec3d;
typedef std::chrono::system_clock::time_point TimePoint;

class CGaussianPuff{
protected:
    const Vector X, Y, Z;
    Vector X_centered, Y_centered;
    int N_points;

    double sim_dt, puff_dt;
    double puff_duration;
    int n_puffs;

    VectorXi hours;
    Vector daylight_times;
    int step_counter = 0;

    const Vector wind_speeds, wind_directions;

    double thresh_xy_max, thresh_z_max;

    Matrix source_coordinates;
    Vector emission_strengths;

    double x0, y0, z0; // current iteration's source coordinates
    double x_min, y_min; // current mins for the grid centered at the current source
    double x_max, y_max; 

    std::function<double(double)> exp;

    double conversion_factor;
    double exp_tol;

    bool skip_low_wind;
    float low_wind_thresh;

    bool quiet = true;

    std::vector<double> lowwind_coefs;

    const double one_over_two_pi_three_halves = 1/std::pow(2*M_PI, 1.5);
    double cosine; // store value of cosine/sine so we don't have to evaluate it across different functions
    double sine;

    std::map<double,int> coef_dists_map = {{0.1, 0}, {0.15, 1}, {0.2, 2}, {0.25, 3}, {0.3, 4}, {0.4, 5}, {0.5, 6}, {0.7, 7}, {1, 8}, {2, 9}, {3, 10}, {3.11, 11}, {4, 12}, {7, 13}, {10, 14}, {15, 15}, {20, 16}, {30, 17}, {40, 18}, {60, 19}, {__INT_MAX__, 20}};
    //stability classes of 'A' and 'G' do not define a value for a,b above x>3.11, define a,b to be -1 arbitrarly, must check on them before use of sigma results will be very low
    std::map<char,std::vector<std::vector<double>>> coefs_map = {
        {'A',{{122.8,0.9447,24.167,2.5334},{158.08,1.0542,24.167,2.5334},{170.22,1.0932,24.167,2.5334},{179.52,1.1262,24.167,2.5334},{217.41,1.2644,24.167,2.5334},{258.89,1.4094,24.167,2.5334},{346.75,1.7283,24.167,2.5334},{453.85,2.1166,24.167,2.5334},{453.85,2.1166,24.167,2.5334},{453.85,2.1166,24.167,2.5334},{453.85,2.1166,24.167,2.5334},{453.85,2.1166,24.167,2.5334},{-1.0,-1.0,24.167,2.5334},{-1.0,-1.0,24.167,2.5334},{-1.0,-1.0,24.167,2.5334},{-1.0,-1.0,24.167,2.5334},{-1.0,-1.0,24.167,2.5334},{-1.0,-1.0,24.167,2.5334},{-1.0,-1.0,24.167,2.5334},{-1.0,-1.0,24.167,2.5334},{-1.0,-1.0,24.167,2.5334}}},
        {'G',{{106.7365,0.93834,21.25,2.1715},{124.3765,0.99309,21.25,2.1715},{130.4465,1.01259,21.25,2.1715},{139.00150000000002,1.05476,21.25,2.1715},{157.9465,1.12386,21.25,2.1715},{178.6865,1.19636,21.25,2.1715},{228.025,1.4127,21.25,2.1715},{281.575,1.60685,21.25,2.1715},{281.575,1.60685,21.25,2.1715},{281.575,1.60685,21.25,2.1715},{281.575,1.60685,21.25,2.1715},{281.575,1.60685,21.25,2.1715},{-1.0,-1.0,21.25,2.1715},{-1.0,-1.0,21.25,2.1715},{-1.0,-1.0,21.25,2.1715},{-1.0,-1.0,21.25,2.1715},{-1.0,-1.0,21.25,2.1715},{-1.0,-1.0,21.25,2.1715},{-1.0,-1.0,21.25,2.1715},{-1.0,-1.0,21.25,2.1715},{-1.0,-1.0,21.25,2.1715}}},
        {'B',{{90.673,0.93198,18.333,1.8096},{90.673,0.93198,18.333,1.8096},{90.673,0.93198,18.333,1.8096},{98.483,0.98332,18.333,1.8096},{98.483,0.98332,18.333,1.8096},{98.483,0.98332,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096},{109.3,1.0971,18.333,1.8096}}},
        {'H',{{75.907,0.923315,15.4165,1.44765},{75.907,0.923315,15.4165,1.44765},{75.907,0.923315,15.4165,1.44765},{79.812,0.948985,15.4165,1.44765},{79.812,0.948985,15.4165,1.44765},{79.812,0.948985,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765},{85.2205,1.005875,15.4165,1.44765}}},
        {'C',{{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857},{61.141,0.91465,12.5,1.0857}}},
        {'I',{{47.8,0.892195,10.4165,0.90476},{47.8,0.892195,10.4165,0.90476},{47.8,0.892195,10.4165,0.90476},{47.8,0.892195,10.4165,0.90476},{47.8,0.892195,10.4165,0.90476},{46.617,0.862655,10.4165,0.90476},{46.617,0.862655,10.4165,0.90476},{46.617,0.862655,10.4165,0.90476},{46.617,0.862655,10.4165,0.90476},{46.617,0.7793399999999999,10.4165,0.90476},{46.617,0.7793399999999999,10.4165,0.90476},{47.3225,0.759755,10.4165,0.90476},{47.3225,0.759755,10.4165,0.90476},{47.3225,0.759755,10.4165,0.90476},{47.3225,0.759755,10.4165,0.90476},{48.8955,0.74027,10.4165,0.90476},{48.8955,0.74027,10.4165,0.90476},{48.8955,0.74027,10.4165,0.90476},{52.597,0.71322,10.4165,0.90476},{52.597,0.71322,10.4165,0.90476},{52.597,0.71322,10.4165,0.90476}}},
        {'D',{{34.459,0.86974,8.333,0.72382},{34.459,0.86974,8.333,0.72382},{34.459,0.86974,8.333,0.72382},{34.459,0.86974,8.333,0.72382},{34.459,0.86974,8.333,0.72382},{32.093,0.81066,8.333,0.72382},{32.093,0.81066,8.333,0.72382},{32.093,0.81066,8.333,0.72382},{32.093,0.81066,8.333,0.72382},{32.093,0.64403,8.333,0.72382},{32.093,0.64403,8.333,0.72382},{33.504,0.60486,8.333,0.72382},{33.504,0.60486,8.333,0.72382},{33.504,0.60486,8.333,0.72382},{33.504,0.60486,8.333,0.72382},{36.65,0.56589,8.333,0.72382},{36.65,0.56589,8.333,0.72382},{36.65,0.56589,8.333,0.72382},{44.053,0.51179,8.333,0.72382},{44.053,0.51179,8.333,0.72382},{44.053,0.51179,8.333,0.72382}}},
        {'J',{{29.359500000000004,0.85317,7.2915,0.633345},{28.895000000000003,0.8446499999999999,7.2915,0.633345},{28.895000000000003,0.8446499999999999,7.2915,0.633345},{28.895000000000003,0.8446499999999999,7.2915,0.633345},{28.895000000000003,0.8446499999999999,7.2915,0.633345},{26.8605,0.78363,7.2915,0.633345},{26.8605,0.78363,7.2915,0.633345},{26.8605,0.78363,7.2915,0.633345},{26.8605,0.78363,7.2915,0.633345},{26.8605,0.6374,7.2915,0.633345},{27.3135,0.607785,7.2915,0.633345},{28.019,0.5882000000000001,7.2915,0.633345},{28.019,0.5882000000000001,7.2915,0.633345},{29.1035,0.5550649999999999,7.2915,0.633345},{29.1035,0.5550649999999999,7.2915,0.633345},{31.81,0.51381,7.2915,0.633345},{31.81,0.51381,7.2915,0.633345},{36.035,0.47102,7.2915,0.633345},{39.7365,0.44397,7.2915,0.633345},{45.8355,0.4038549999999999,7.2915,0.633345},{45.8355,0.4038549999999999,7.2915,0.633345}}},
        {'E',{{24.26,0.8366,6.25,0.54287},{23.331,0.81956,6.25,0.54287},{23.331,0.81956,6.25,0.54287},{23.331,0.81956,6.25,0.54287},{23.331,0.81956,6.25,0.54287},{21.628,0.7566,6.25,0.54287},{21.628,0.7566,6.25,0.54287},{21.628,0.7566,6.25,0.54287},{21.628,0.7566,6.25,0.54287},{21.628,0.63077,6.25,0.54287},{22.534,0.57154,6.25,0.54287},{22.534,0.57154,6.25,0.54287},{22.534,0.57154,6.25,0.54287},{24.703,0.50527,6.25,0.54287},{24.703,0.50527,6.25,0.54287},{26.97,0.46173,6.25,0.54287},{26.97,0.46173,6.25,0.54287},{35.42,0.37615,6.25,0.54287},{35.42,0.37615,6.25,0.54287},{47.618,0.29592,6.25,0.54287},{47.618,0.29592,6.25,0.54287}}},
        {'K',{{19.7345,0.82609,5.208349999999999,0.4523899999999999},{19.27,0.8175699999999999,5.208349999999999,0.4523899999999999},{19.27,0.8175699999999999,5.208349999999999,0.4523899999999999},{18.894,0.8018149999999999,5.208349999999999,0.4523899999999999},{18.894,0.8018149999999999,5.208349999999999,0.4523899999999999},{18.0425,0.770335,5.208349999999999,0.4523899999999999},{18.0425,0.770335,5.208349999999999,0.4523899999999999},{18.0425,0.770335,5.208349999999999,0.4523899999999999},{17.7905,0.7206250000000001,5.208349999999999,0.4523899999999999},{17.7905,0.6315200000000001,5.208349999999999,0.4523899999999999},{18.6785,0.558285,5.208349999999999,0.4523899999999999},{19.3605,0.51822,5.208349999999999,0.4523899999999999},{19.3605,0.51822,5.208349999999999,0.4523899999999999},{20.445,0.485085,5.208349999999999,0.4523899999999999},{21.2695,0.4601699999999999,5.208349999999999,0.4523899999999999},{22.403,0.4384,5.208349999999999,0.4523899999999999},{24.8105,0.39427,5.208349999999999,0.4523899999999999},{29.0355,0.35148,5.208349999999999,0.4523899999999999},{31.247,0.3252549999999999,5.208349999999999,0.4523899999999999},{37.346,0.28514,5.208349999999999,0.4523899999999999},{37.346,0.28514,5.208349999999999,0.4523899999999999}}},
        {'F',{{15.209,0.81558,4.1667,0.36191},{15.209,0.81558,4.1667,0.36191},{15.209,0.81558,4.1667,0.36191},{14.457,0.78407,4.1667,0.36191},{14.457,0.78407,4.1667,0.36191},{14.457,0.78407,4.1667,0.36191},{14.457,0.78407,4.1667,0.36191},{14.457,0.78407,4.1667,0.36191},{13.953,0.68465,4.1667,0.36191},{13.953,0.63227,4.1667,0.36191},{14.823,0.54503,4.1667,0.36191},{16.187,0.4649,4.1667,0.36191},{16.187,0.4649,4.1667,0.36191},{16.187,0.4649,4.1667,0.36191},{17.836,0.41507,4.1667,0.36191},{17.836,0.41507,4.1667,0.36191},{22.651,0.32681,4.1667,0.36191},{22.651,0.32681,4.1667,0.36191},{27.074,0.27436,4.1667,0.36191},{27.074,0.27436,4.1667,0.36191},{27.074,0.27436,4.1667,0.36191}}}
    };

public:
    /* Constructor.
    Inputs:
        X, Y, Z: Flattened vectors holding (x,y,z) coordinates of points to simulate [m]
        N: total number of grid points
        sim_dt: time between simulation time steps [s]
        puff_dt: time between creation of two puffs [s]
        puff_duration: maximum amount of time a puff can be alive for. Helps prevent unphysical effects [s]
        sim_start, sim_end: datetime stamps for when to start and end the emission scenario
        wind_speeds, wind_directions: timeseries for wind speeds [m/s] and directions [degrees] at sim_dt resolution
        source_coordinates: source coordinates in (x,y,z) format for each source. size- (n_sources, 3) [m]
        emission_strengths: emission rates for each source [kg/hr]. length: n_sources
        conversion_factor: conversion ratio between kg/m^3 to ppm for CH4
        exp_tol: tolerance for the exponential thresholding applied to the Gaussians. Lower tolerance means less accuracy
        but a faster execution.
        unsafe: enables unsafe math operations that are faster but less accurate due to the approximations used.
        quiet: false if you want output for simulation completeness. true for silent simulation.
    */
  CGaussianPuff(Vector X, Vector Y, Vector Z, int N, double sim_dt,
                double puff_dt, double puff_duration, int n_puffs,
                VectorXi hours, Vector wind_speeds, Vector wind_directions,
                Matrix source_coordinates, Vector emission_strengths,
                double conversion_factor, double exp_tol, bool skip_low_wind,
                float low_wind_thresh, bool unsafe, bool quiet, Vector daylight_times, std::vector<double> lowwind_coefs)

      : X(X), Y(Y), Z(Z), N_points(N), sim_dt(sim_dt), puff_dt(puff_dt),
        puff_duration(puff_duration), n_puffs(n_puffs), hours(hours),
        wind_speeds(wind_speeds), wind_directions(wind_directions),
        source_coordinates(source_coordinates),
        emission_strengths(emission_strengths),
        conversion_factor(conversion_factor), exp_tol(exp_tol),
        skip_low_wind(skip_low_wind), low_wind_thresh(low_wind_thresh),
        quiet(quiet), daylight_times(daylight_times), lowwind_coefs(lowwind_coefs) {

        
    if (unsafe) {
      if (!quiet)
        std::cout << "RUNNING IN UNSAFE MODE\n";
      this->exp = &fastExp;
    } else {
      this->exp = [](double x) { return std::exp(x); };
    }

}

    /* Simulation time loop
    Inputs:
        ch4: 2d array. First index represents simulation time steps, second index is flattened spatial index
    Returns:
        none, but the concentration is added directly to the ch4 array for all time steps
    */
    void simulate(RefMatrix ch4){

        // later, for multisource: iterate over source coords
        setSourceCoordinates(0);
        double q = emission_strengths[0]/3600; // convert to kg/s

        double emission_per_puff = q*puff_dt;

        double report_ratio = 0.1;

        int puff_lifetime = ceil(puff_duration/sim_dt); // number of time steps a puff can live for
        int puff_to_sim_ratio = round(puff_dt/sim_dt); // number of simulation steps for every puff

        for(int p = 0; p < n_puffs; p++){
          if (skip_low_wind && wind_speeds[p] < low_wind_thresh)
            continue;

          if (PyErr_CheckSignals() != 0)
            throw pybind11::error_already_set(); // catches ctrl+c signal from
                                                 // python

          // bounds check on time
          if (p * puff_to_sim_ratio + puff_lifetime >= ch4.rows())
            puff_lifetime = ch4.rows() - p * puff_to_sim_ratio;

          double theta = windDirectionToAngle(wind_directions[p]);

          // computes concentration timeseries for this puff
          concentrationPerPuff(
              emission_per_puff, theta, wind_speeds[p], hours[p],
              ch4.middleRows(p * puff_to_sim_ratio, puff_lifetime));

          if (!quiet && floor(n_puffs * report_ratio) == p) {
            std::cout << "Simulation is " << report_ratio * 100 << "\% done\n";
            report_ratio += 0.1;
          }
        }
    }

private:

    /* Computes the concentration timeseries for a single puff.
    Inputs:
        q: Total emission corresponding to this puff (kg)
        ws, theta: wind speed (m/s) and wind direction (radians)
        hour: current hour of day (int)
        ch4: 2D concentration array. First index is time, second index is the flattened spatial index.
    Returns:
        None. The concentration is added directly into the ch4 array in GaussianPuffEquation()
    */
    void concentrationPerPuff(double q, double theta, double ws, int hour,
                                RefMatrix ch4){

        // cache cos/sin so they can get reused in other calls
        cosine = cos(theta);
        sine = sin(theta);


        std::vector<char> stability_class = stabilityClassifier(ws, hour);

        GaussianPuffEquation(q, ws,stability_class,ch4);
    }

    void getPuffCenteredSigmas(std::vector<double>& sigma_y, std::vector<double>& sigma_z,
                                         int n_time_steps, double ws, std::vector<char> stability_class){
        std::vector<double> downwind_dists = std::vector<double>(n_time_steps+1);

        double shift_per_step = ws*sim_dt;
        downwind_dists[0] = 0;
        for(int i = 1; i <= n_time_steps; i++){
            downwind_dists[i] = downwind_dists[i-1] + shift_per_step;
        }

        getSigmaCoefficients(sigma_y, sigma_z, stability_class, downwind_dists);
    }

    Vector2d calculateExitLocation(){

        Vector2d box_min(x_min, y_min);
        Vector2d box_max(x_max, y_max);
        Vector2d origin(0,0);
        Vector2d rayDir(cosine, -sine);
        Vector2d invRayDir = rayDir.cwiseInverse();

        Vector2d exit_times = AABB(box_min, box_max, origin, invRayDir);

        return exit_times[1]*rayDir;
    }

    /* Evaluates the Gaussian Puff equation on the grids. 
    Inputs:
        q: Total emission corresponding to this puff (kg)
        ws, wind speed (m/s)
        X_rot, Y_rot: rotated X and Y grids. The Z grid isn't rotated so the member variable is used repeatedly.
        ts: time series the puff is live for. 
        c: 2D concentration array. The first index represents the time step, the second index represents the flattened
        spatial index.
    Returns:
        none, but the concentrations are added into the concentration array.
    */
    void GaussianPuffEquation(
        double q, double ws, std::vector<char> stability_class,
        RefMatrix ch4) {

        Vector2d exit_location = calculateExitLocation();
        double max_downwind_dist = sqrt(exit_location[0]*exit_location[0] + exit_location[1]*exit_location[1]);

        std::vector<double> temp(1);
        std::vector<double> temp_y(1);
        std::vector<double> temp_z(1);
        temp[0] = max_downwind_dist;

        getSigmaCoefficients(temp_y, temp_z, stability_class, temp); // get maximum sigma coeffs
        double sigma_y_max = temp_y[0];
        double sigma_z_max = temp_z[0];


        // compute the maximum plume size
        double prefactor = (q * conversion_factor * one_over_two_pi_three_halves) / (sigma_y_max * sigma_y_max * sigma_z_max);
        double threshold = std::log(exp_tol / (2 * prefactor));
        double thresh_constant = std::sqrt(-2 * threshold);

        thresh_xy_max = sigma_y_max * thresh_constant;
        thresh_z_max = sigma_z_max * thresh_constant;

        // find out when puff will leave the domain
        double t = calculatePlumeTravelTime(thresh_xy_max, ws); // number of seconds til plume leaves grid

        int n_time_steps = ceil(t / sim_dt); // rescale to unitless number of timesteps

        std::vector<double> sigma_y(n_time_steps+1);
        std::vector<double> sigma_z(n_time_steps+1);

        // gets the dispersion coefficient for the puff at each location on its path
        getPuffCenteredSigmas(sigma_y, sigma_z, n_time_steps, ws, stability_class);

        // bound check on time
        if (n_time_steps >= ch4.rows()) {
            n_time_steps = ch4.rows() - 1;
        }

        double shift_per_step = ws*sim_dt;
        double x_shift_per_step = cosine*shift_per_step;
        double y_shift_per_step = -sine*shift_per_step;
        double wind_shift  = 0;
        double x_shift = 0;
        double y_shift = 0;

        // prefactor excluding the sigmas since those change each time step
        prefactor = q*conversion_factor*one_over_two_pi_three_halves;

        // start at 1 since plume hasn't moved any on it's 0th time step so it's effective dispersion is 0
        for (int i = 1; i <= n_time_steps; i++) {
            // recompute threshold we use every step with current plume dispersion coefficient
            double one_over_sig_y = 1/sigma_y[i];
            double one_over_sig_z = 1/sigma_z[i]; 
            double local_prefactor = prefactor*one_over_sig_y*one_over_sig_y*one_over_sig_z;

            double temp = std::log(exp_tol / (2 * local_prefactor));
            double local_thresh = std::sqrt(-2 * temp);

            // wind_shift is distance [m] plume has moved from source
            wind_shift += shift_per_step;
            x_shift += x_shift_per_step;
            y_shift += y_shift_per_step;

            std::vector<int> indices = coarseSpatialThreshold(wind_shift, local_thresh, sigma_y[i], sigma_z[i]);
 
            for (int j : indices) {

                // Skips upwind cells since upwind diffusion is ignored
                if(sigma_y[i] <= 0 || sigma_z[i] <= 0){
                    continue;
                }

                double t_xy = sigma_y[i] * local_thresh; // local threshold

                // Exponential thresholding conditionals
                if (std::abs(X_centered[j] - x_shift) >= t_xy) {
                    continue;
                }

                if (std::abs(Y_centered[j] - y_shift) >= t_xy) {
                    continue;
                }

                double t_z = sigma_z[i] * local_thresh; // local threshold

                if (std::abs(Z[j] - z0) >= t_z) {
                    continue;
                }

                // terms are written in a way to minimize divisions and exp evaluations
                double y_dist_from_cent = (Y_centered[j] - y_shift);
                double x_dist_from_cent = (X_centered[j] - x_shift);
                double z_minus_by_sig = (Z[j] - z0) * one_over_sig_z;
                double z_plus_by_sig = (Z[j] + z0) * one_over_sig_z;

                double one_over_sig_y_sq = one_over_sig_y*one_over_sig_y;

                double term_4_a_arg = z_minus_by_sig * z_minus_by_sig;
                double term_4_b_arg = z_plus_by_sig * z_plus_by_sig;
                double term_3_arg = (y_dist_from_cent * y_dist_from_cent + x_dist_from_cent * x_dist_from_cent)*one_over_sig_y_sq;

                double term_4 = this->exp(-0.5 * (term_3_arg + term_4_a_arg)) + this->exp(-0.5 * (term_3_arg + term_4_b_arg));

                ch4(i, j) += local_prefactor * term_4;
            }
        }
    }

    /* Computes the time step when the plume will exit the computational grid. 
    Inputs:
        thresh_xy: Gaussian threshold on xy
        ws, wind speed (m/s)
    Returns:
        the time when the plume will be fully off the grid.
    */
    double calculatePlumeTravelTime(double thresh_xy, 
                                    double ws){

        Vector2d box_min(-thresh_xy, -thresh_xy);
        Vector2d box_max(thresh_xy, thresh_xy);

        Vector2d grid_min(x_min, y_min);
        Vector2d grid_max(x_max, y_max);

        Vector2d origin(0,0);

        Vector2d rayDir(cosine, -sine);
        Vector2d invRayDir = rayDir.cwiseInverse();

        // finding the last corner of the threshold box to leave the grid
        Vector2d box_times = AABB(box_min, box_max, origin, invRayDir); // find where ray intersects box
        Vector2d backward_collision = box_times[0]*rayDir; // where backwards ray intersects with an edge of the box
        Vector2d box_corner = findNearestCorner(box_min, box_max, backward_collision);

        // find the corner of the grid that the threshold must pass based on the wind direction
        Vector2d grid_middle = 0.5*(grid_max-grid_min).array() + grid_min.array();
        Vector2d grid_times = AABB(grid_min, grid_max, grid_middle, invRayDir);
        Vector2d forward_collision = grid_times[1]*rayDir + grid_middle;
        Vector2d grid_corner = findNearestCorner(grid_min, grid_max, forward_collision);

        // compute travel time between the two corners
        Vector2d distance = (grid_corner-box_corner).cwiseAbs();
        invRayDir = invRayDir.cwiseAbs();
        double travelDistance = (distance.array()*invRayDir.array()).minCoeff();

        double travelTime = travelDistance/ws;

        return travelTime;
    }

    /* Axis Aligned Bounding Box algorithm. Used to compute the intersections between a ray (wind direction) and a square.
    See https://tavianator.com/2022/ray_box_boundary.html for details.
    Inputs:
        box_min, box_max: 2D vectors containing the minimum and maximum corners of a square in the xy plane.
        origin: starting location of the ray.
        invRayDir: elementwise inverse of the ray direction. While the ray direction could be used instead as the input,
            using the inverse saves on computing multiple divisions.
    Returns:
        2D vector containing the times of the ray intersection. For the fast Gaussian Puff algorithm, the ray's origin
        is always within the box. As such, the returns will have tmin < 0 be the backwards intersection and tmax > 0 
        be the forward intersection, where the directions refer to traveling in the positive and negative ray direction.
    */
    Vector AABB(Vector box_min, Vector box_max, Vector origin, Vector invRayDir){

        // casts to arrays make it an elementwise product
        Vector t0 = (box_min-origin).array()*invRayDir.array();
        Vector t1 = (box_max-origin).array()*invRayDir.array();

        double tmax = (t0.cwiseMax(t1)).minCoeff();
        double tmin = (t0.cwiseMin(t1)).maxCoeff();

        return Vector2d(tmin, tmax);
    }
 
    /*  Given a point on the edge of a square, finds the nearest of the four corners of the square.
    Inputs:
        min_corner, max_corner:  minimum and maximum corners of a square (e.g. lower left and top right).
        point: a point on the edge of the square.
    Returns:
        2D Vector containing the coordinates to the nearest corner of the square. 
    */
    Vector findNearestCorner(Vector min_corner, Vector max_corner, Vector point){
        
        Vector2d corner;

        if(abs(min_corner(0)-point(0)) < abs(max_corner(0)-point(0))){
            corner(0) = min_corner(0);
        }else{
            corner(0) = max_corner(0);
        }

        if(abs(min_corner(1)-point(1)) < abs(max_corner(1)-point(1))){
            corner(1) = min_corner(1);
        }else{
            corner(1) = max_corner(1);
        }

        return corner;
    }

    /* Computes Pasquill stability class
    Inputs:
        wind_speed: [m/s]
        hour: current hour of day
    Returns:
        stability_class: character A-F representing a Pasquill stability class
            note: multiple classes can be returned. In this case, they get averaged when computing dispersion coefficients.
    */


    std::vector<char> stabilityClassifier(double wind_speed, int hour) {

        bool is_day = daylight_times[step_counter];
        step_counter++;
        // bool is_day = (hour >= 7) && (hour <= 18);
        if (wind_speed < 2) {
            return is_day ? std::vector<char>{'A', 'B'} : std::vector<char>{'K'};
        } else if (wind_speed < 3) {
            return is_day ? std::vector<char>{'B'} : std::vector<char>{'E', 'F'};
        } else if (wind_speed < 5) {
            return is_day ? std::vector<char>{'B', 'C'} : std::vector<char>{'D', 'E'};
        } else if (wind_speed < 6) {
            return is_day ? std::vector<char>{'C', 'D'} : std::vector<char>{'D'};
        } else {
            return std::vector<char>{'D'};
        }
    }


    /* Gets dispersion coefficients (sigma_{y,z}) for the entire grid.
        sigma_z = a*x^b, x in km,
        sigma_y = 465.11628*x*tan(THETA) where THETA = 0.017453293*(c-d*ln(x)) where x in km
        Note: sigma_{y,z} = -1 if x < 0 due to there being no upwind dispersion.
    Inputs:
        sigma_y, sigma_z: vectors to fill with dispersion coefficients
        stability_class: a char in A-F from Pasquill stability classes 
        X_rot: rotated version of the X grid
    Returns:
        None, but sigma_y and sigma_z are filled with the dispersion coefficients.
    */
    void getSigmaCoefficients(std::vector<double>& sigma_y, std::vector<double>& sigma_z, std::vector<char> stability_class, std::vector<double>& downwind_dists){
        int n_stab = stability_class.size();

        for(int i = 0; i < downwind_dists.size(); i++){

            double x = downwind_dists[i]*0.001;

            double sigma_y_temp;
            double sigma_z_temp;

            // note: if there are multiple stability classes, we average the dispersion coefficients
            for(int j = 0; j < n_stab; j++){
                char stab = stability_class[j];
                int flag = 0;
                double a, b, c, d;

            if (x <= 0) {
                sigma_y[i] = -1;
                sigma_z[i] = -1;
                break;
                
            } else if ((stab == 'K') && (x < 0.1) && (lowwind_coefs.size() > 0)) {
                a = lowwind_coefs[0];
                b = lowwind_coefs[1];
                c = lowwind_coefs[2];
                d = lowwind_coefs[3];

            }else {
                for (const auto& pair : coef_dists_map){
                    if (x < pair.first){
                        a = coefs_map[stab][pair.second][0];
                        b = coefs_map[stab][pair.second][1];
                        c = coefs_map[stab][pair.second][2];
                        d = coefs_map[stab][pair.second][3];
                        break;
                    }
                }
                //a,b not defined for 'A','G' classes above x>0.1, assigned -1 arbitratly in coefs map
                if ((a < 0) || (b < 0)){
                    flag=1;
                }
            }
                double Theta = 0.017453293 * (c - d * std::log(x)); // in radians
                sigma_y_temp = 465.11628 * x * std::tan(Theta); // in meters
    
                if (flag == 0) {
                    sigma_z_temp = a * std::pow(x, b); // in meters
                    sigma_z_temp = std::min(sigma_z_temp, 5000.0);
                } else {
                    sigma_z_temp = 5000.0;
                }
                // }
                sigma_y[i] += sigma_y_temp;
                sigma_z[i] += sigma_z_temp;

            }
            sigma_y[i] /= n_stab;
            sigma_z[i] /= n_stab;
        }
    }


    const double deg_to_rad_factor = M_PI/180.0;

    // somewhat hacky way of making this act as an abstract function
    // okay with this for now since it's private and can't get called from python due to lack of bindings
    virtual std::vector<int> coarseSpatialThreshold(double, double, double, double){ throw std::logic_error("Not implemented"); }

    static double fastExp(double x){
        constexpr double a = (1ll << 52) / 0.6931471805599453;
        constexpr double b = (1ll << 52) * (1023 - 0.04367744890362246);
        x = a * x + b;

        constexpr double c = (1ll << 52);
        if (x < c)
            x = 0.0;

        uint64_t n = static_cast<uint64_t>(x);
        std::memcpy(&x, &n, 8);
        return x;
    }

    void setSourceCoordinates(int source_index){
        x0 = source_coordinates(source_index, 0);
        y0 = source_coordinates(source_index, 1);
        z0 = source_coordinates(source_index, 2);

        // center the x,y grids at the source
        X_centered = X.array() - x0;
        Y_centered = Y.array() - y0;

        x_min = X.minCoeff() - x0;
        y_min = Y.minCoeff() - y0;
        x_max = X.maxCoeff() - x0;
        y_max = Y.maxCoeff() - y0;
    }

    // convert wind direction (degrees) to the angle (radians) between the wind vector and positive x-axis
    double windDirectionToAngle(double wd){
        double theta = wd-270;
        theta = theta*deg_to_rad_factor;

        return theta;
    }

};

class GridGaussianPuff : public CGaussianPuff {

    int nx, ny, nz;
    double dx, dy, dz; // grid spacings
public:

    /* Constructor. See CGaussianPuff constructor for information on parameters not mentioned here.
    nx,ny,nz: number of grid points along each axis respectively
    */
  GridGaussianPuff(Vector X, Vector Y, Vector Z, int nx, int ny, int nz,
                   double sim_dt, double puff_dt, double puff_duration,
                   int n_puffs, VectorXi hours, Vector wind_speeds,
                   Vector wind_directions, Matrix source_coordinates,
                   Vector emission_strengths, double conversion_factor,
                   double exp_tol, bool skip_low_wind, float low_wind_thresh,
                   bool unsafe, bool quiet, Vector daylight_times, std::vector<double> lowwind_coefs)

      : CGaussianPuff(X, Y, Z, nx * ny * nz, sim_dt, puff_dt, puff_duration,
                      n_puffs, hours, wind_speeds, wind_directions,
                      source_coordinates, emission_strengths, conversion_factor,
                      exp_tol, skip_low_wind, low_wind_thresh, unsafe, quiet, daylight_times, lowwind_coefs),
        nx(nx), ny(ny), nz(nz) {

    std::vector<double> gridSpacing = computeGridSpacing();
    dx = gridSpacing[0];
    dy = gridSpacing[1];
    dz = gridSpacing[2];

    // declares empty 3D vector of integers of size (nx, ny, nz)
    vec3d map_table(ny,
                    std::vector<std::vector<int>>(nx, std::vector<int>(nz)));

    // precomputes the map from the 3D meshgrid index to the 1D raveled index.
    // precomputed because the divisions in map() are too expensive to do
    // repeatedly.
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
        for (int k = 0; k < nz; k++) {
          // (i,j) index flipped since numpy's 'ij' indexing is being used on
          // the meshgrids
          map_table[j][i][k] = map(i, j, k);
        }
      }
    }
    this->map_table = map_table;
  }

private:
    vec3d map_table; // precomputed map from the 3D meshgrid index to the 1D raveled index.
    

    // The grid-based spatial thresholding uses inequalities based on grid indices 
    std::vector<int> coarseSpatialThreshold(double wind_shift, double thresh_constant, 
                        double sig_y_current,  double sig_z_current) override {

        std::vector<int> indices = getValidIndices(thresh_xy_max, thresh_z_max, wind_shift);

        return indices;
    }

    /*  Computes bounds on the grid indices based on where the Gaussian is located.
    Inputs:
        thresh_xy, thresh_z: Gaussian thresholds on x and y together, and z separately. These are based on the
            dispersion coefficients of the Gaussian. For the loosest possible bounds, use the largest coefficients.
        ws, wind speed (m/s)
        t_i: time step (s)

    Returns:
        A vector of six doubles containing the lower and upper bounds on the i, j, and k indices. Note that these are
        not rounded to integers as they're used in an intermediate calculation (see calculatePlumeTravelTime) and
        roundind them early creates a rounding error.
    */
    std::vector<double> computeIndexBounds(double thresh_xy, double thresh_z,
                                            double wind_shift){

        Eigen::Matrix2d R;
        R << cosine, -sine,
            sine, cosine;

        Eigen::Vector2d X0;
        X0 << x_min, y_min;

        Eigen::Vector2d v = R.col(0);
        Eigen::Vector2d vp = R.col(1);

        Eigen::Vector2d tw;
        tw << wind_shift, 0;

        Eigen::Vector2d X0_r = R*X0;
        auto X0_rt = X0_r - tw;

        double Xrt_dot_v = X0_rt.dot(v);
        double Xrt_dot_vp = X0_rt.dot(vp);

        double one_over_dx = 1/dx;
        double one_over_dy = 1/dy;
        double one_over_dz = 1/dz;

        double i_lower = (-Xrt_dot_v - thresh_xy - 1)*one_over_dx;
        double i_upper = (-Xrt_dot_v + thresh_xy + 1)*one_over_dx;

        double j_lower = (-Xrt_dot_vp - thresh_xy - 1)*one_over_dy;
        double j_upper = (-Xrt_dot_vp + thresh_xy + 1)*one_over_dy;

        double k_lower = (-thresh_z + z0)*one_over_dz;
        double k_upper = (thresh_z + z0)*one_over_dz;

        return std::vector<double>{i_lower, i_upper, j_lower, j_upper, k_lower, k_upper};
    }

    /* Computes a list of indices to be evaluated based on the location of the Gaussian on the grid.
    Inputs:
        thresh_xy, thresh_z: Gaussian thresholds on x and y together, and z separately. These are based on the
            dispersion coefficients of the Gaussian. For the loosest possible bounds, use the largest coefficients.
        wind_shift: meters that the plume is shifted downwind at the current timestep
        x0, y0, z0: coordinates of source (m)
    Returns:
        A list of indices to the flattened grids where the Gaussian equation should be evaluated.
    */
    std::vector<int> getValidIndices(double thresh_xy, double thresh_z,
                                        double wind_shift){
        
        std::vector<double> indexBounds = computeIndexBounds(thresh_xy, thresh_z,
                                                            wind_shift);

        int i_lower = floor(indexBounds[0]);
        int i_upper = ceil(indexBounds[1]);
        int j_lower = floor(indexBounds[2]);
        int j_upper = ceil(indexBounds[3]);
        int k_lower = floor(indexBounds[4]);
        int k_upper = ceil(indexBounds[5]);

        // makes sure the computed index bounds are sensical and computes total number of cells in bounds
        if(i_lower < 0) i_lower = 0;
        if(i_upper > nx-1) i_upper = nx-1;

        if(j_lower < 0) j_lower = 0;
        if(j_upper > ny-1) j_upper = ny-1;

        if(k_lower < 0) k_lower = 0;
        if(k_upper > nz-1) k_upper = nz-1;

        int i_count, j_count, k_count;
        if(i_upper < i_lower || i_lower > i_upper){
            return std::vector<int>(0);
        } else{
            i_count = i_upper-i_lower+1;
        }

        if(j_upper < j_lower || j_lower > j_upper){
            return std::vector<int>(0);
        } else{
            j_count = j_upper-j_lower+1;
        }

        if(k_upper < k_lower){
            return std::vector<int>(0);
        } else{
            k_count = k_upper-k_lower+1;
        }

        int cellCount = i_count*j_count*k_count;

        std::vector<int> indices(cellCount);
        int currentCell = 0;
        for(int i = i_lower; i <= i_upper; i++){
            for(int j = j_lower; j <= j_upper; j++){
                for(int k = k_lower; k <= k_upper; k++){
                    indices[currentCell] = map_table[j][i][k];
                    currentCell++;
                }
            }
        }

        return indices;
    }

    std::vector<double> computeGridSpacing(){

        std::vector<double> gridSpacing(3); 

        gridSpacing[0] = abs(X[nz] - X[0]); // dx
        gridSpacing[1] = abs(Y[nz*nx] - Y[0]); // dy
        gridSpacing[2] = abs(Z[1] - Z[0]); // dz

        return gridSpacing;
    }

    // maps 3d index to 1d raveled index in numpy 'ij' format meshgrids
    int map(int i, int j, int k){
        return j*nz*nx + i*nz + k;
    }
    
};

class SensorGaussianPuff : public CGaussianPuff {

    std::vector<int> indices;
    
public:

    /* Constructor. See CGaussianPuff constructor for information about parameters not given here.
    N_sensors: Number of sensors being simulated.
    */
  SensorGaussianPuff(Vector X, Vector Y, Vector Z, int N_sensors, double sim_dt,
                     double puff_dt, double puff_duration, int n_puffs,
                     VectorXi hours, Vector wind_speeds, Vector wind_directions,
                     Matrix source_coordinates, Vector emission_strengths,
                     double conversion_factor, double exp_tol,
                     bool skip_low_wind, float low_wind_thresh, bool unsafe,
                     bool quiet, Vector daylight_times, std::vector<double> lowwind_coefs)
      : CGaussianPuff(X, Y, Z, N_sensors, sim_dt, puff_dt, puff_duration,
                      n_puffs, hours, wind_speeds, wind_directions,
                      source_coordinates, emission_strengths, conversion_factor,
                      exp_tol, skip_low_wind, low_wind_thresh, unsafe, quiet, daylight_times, lowwind_coefs) {

    std::vector<int> inds(N_sensors);
    for (int i = 0; i < N_sensors; i++) {
      inds[i] = i;
    }

    this->indices = inds;
  }

private:
    // mostly a stub, can add a precomputed spatial threshold later
    std::vector<int> coarseSpatialThreshold(double wind_shift, double thresh_constant, 
                        double sig_y_current,  double sig_z_current) override {
        return indices;
    }

};

using namespace pybind11::literals;
namespace py = pybind11;

PYBIND11_MODULE(CGaussianPuff, m) {
  py::class_<GridGaussianPuff>(m, "GridGaussianPuff")
      .def(py::init<Vector, Vector, Vector, int, int, int, double, double,
                    double, int, VectorXi, Vector, Vector, Matrix, Vector,
                    double, double, bool, float, bool, bool, Vector, std::vector<double>>())
      .def("simulate", &CGaussianPuff::simulate);

  py::class_<SensorGaussianPuff>(m, "SensorGaussianPuff")
      .def(py::init<Vector, Vector, Vector, int, double, double, double, int,
                    VectorXi, Vector, Vector, Matrix, Vector, double, double,
                    bool, double, bool, bool, Vector, std::vector<double>>())
      .def("simulate", &CGaussianPuff::simulate);
}
