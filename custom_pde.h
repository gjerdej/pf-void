// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>

#include <random>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
class CustomPDE : public PDEOperatorBase<dim, degree, number>
{
public:
  using ScalarValue = dealii::VectorizedArray<number>;
  using ScalarGrad  = dealii::Tensor<1, dim, ScalarValue>;
  using ScalarHess  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorValue = dealii::Tensor<1, dim, ScalarValue>;
  using VectorGrad  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorHess  = dealii::Tensor<3, dim, ScalarValue>;
  using PDEOperatorBase<dim, degree, number>::get_user_inputs;
  using PDEOperatorBase<dim, degree, number>::get_pf_tools;

  /**
   * @brief Constructor.
   */
  CustomPDE(const UserInputParameters<dim> &_user_inputs,
            PhaseFieldTools<dim>           &_pf_tools)
    : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools)
    , D_bulk(get_user_inputs().user_constants.get_double("D_bulk"))
    , gamma(get_user_inputs().user_constants.get_double("gamma"))
    , J_app(get_user_inputs().user_constants.get_double("J_app"))
    , c_max(get_user_inputs().user_constants.get_double("c_max"))
    , kappa(0.0)
    , W(0.0)
    , n_int(get_user_inputs().user_constants.get_double("n_int"))
    , dx(0.0)
    , c_interface_width(0.0)
    , psi_interface_width(0.0)
    , E_Li(get_user_inputs().user_constants.get_double("E_Li"))
    , nu_Li(get_user_inputs().user_constants.get_double("nu_Li"))
    , p_stack(get_user_inputs().user_constants.get_double("p_stack"))
    , penalty_factor(get_user_inputs().user_constants.get_double("penalty_factor"))
    , elastic_floor(get_user_inputs().user_constants.get_double("elastic_floor"))
    , y_mech_bottom(get_user_inputs().user_constants.get_double("y_mech_bottom"))
    , stiffness(get_user_inputs().user_constants.get_elasticity_tensor_plane_strain("stiffness"))
  {
    const auto &mesh = get_user_inputs().spatial_discretization.rectangular_mesh;

    const unsigned int refinement = get_user_inputs().spatial_discretization.global_refinement;

    const double refinement_factor = std::ldexp(1.0, refinement); // 2^refinement

    const double hx = mesh.size[0] / (static_cast<double>(mesh.subdivisions[0]) * refinement_factor);

    const double hy = mesh.size[1] / (static_cast<double>(mesh.subdivisions[1]) * refinement_factor);

    dx = std::min(hx, hy);

    // Assumes n_int means the number of element spacings.
    c_interface_width = n_int * dx;

    W = 48.0 * std::log(3) * gamma / c_interface_width;

    kappa = std::sqrt(3.0 * gamma * c_interface_width / (2.0 * std::log(3)));

    // Optional: use the same 0.1–0.9 width for psi.
    psi_interface_width = c_interface_width;
  }

private:
  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override
  {
    const dealii::Tensor<1, dim> &mesh_size       = get_user_inputs().spatial_discretization.rectangular_mesh.size;
    const std::vector<unsigned int> &mesh_subdivs = get_user_inputs().spatial_discretization.rectangular_mesh.subdivisions;

    if (index == 0)
      {
        scalar_value = 1.0;      
      }

    if (index == 3)
      {
        const double hx = mesh_size[0] / static_cast<double>(mesh_subdivs[0]);
        const double hy = mesh_size[1] / static_cast<double>(mesh_subdivs[1]);

        // Use the smaller element spacing to define the interface width.
        const double h = std::min(hx, hy);

        const double x0 = 0.5 * mesh_size[0];
        const double y0 = 0.5 * mesh_size[1];
        const double radius = mesh_size[0] / 6.0;

        // Signed distance to the horizontal substrate.
        // Negative below y = y0.
        const double d_plane = point[1] - y0;

        // Signed distance to the circle centered on the substrate surface.
        // Negative inside the circle.
        const double dx = point[0] - x0;
        const double dy = point[1] - y0;
        const double d_circle = std::sqrt(dx * dx + dy * dy) - radius;

        // Union of substrate and circle.
        const double signed_distance = std::min(d_plane, d_circle);

        /*
        * Define the interface width as the distance over which psi changes
        * from 0.9 to 0.1. For
        *
        *   psi = 0.5 * (1 - tanh(d / epsilon)),
        *
        * the 0.9-to-0.1 width is
        *
        *   2 * atanh(0.8) * epsilon.
        */
        const double epsilon = psi_interface_width / (2.0 * std::log(3));

        const double psi_min = 1.0e-6;

        const double profile = 0.5 * (1.0 - std::tanh(signed_distance / epsilon));

        scalar_value = psi_min + (1.0 - psi_min) * profile;
      }

    if (index == 5)
      {
        const double hx = mesh_size[0] / static_cast<double>(mesh_subdivs[0]);
        const double hy = mesh_size[1] / static_cast<double>(mesh_subdivs[1]);

        // Use the smaller element spacing to define the interface width.
        const double h = std::min(hx, hy);

        const double y0 = 0.9 * mesh_size[1];

        // Signed distance to the horizontal substrate.
        // Negative below y = y0.
        const double d_plane = point[1] - y0;

        // Union of substrate and circle.
        const double signed_distance = d_plane;

        /*
        * Define the interface width as the distance over which psi changes
        * from 0.9 to 0.1. For
        *
        *   psi = 0.5 * (1 - tanh(d / epsilon)),
        *
        * the 0.9-to-0.1 width is
        *
        *   2 * atanh(0.8) * epsilon.
        */
        const double epsilon = psi_interface_width / (2.0 * std::log(3));

        const double psi_min = 1.0e-6;

        const double profile = 0.5 * (1.0 - std::tanh(signed_distance / epsilon));

        scalar_value = psi_min + (1.0 - psi_min) * profile;
      }

    if (index == 6) // Vector displacement u
      {
        vector_component_value = 0.0;
      }

    if (index == 7) // Mechanical bottom indicator
      {
        const double epsilon = psi_interface_width / (2.0 * std::log(3.0));

        // One above the bottom boundary; zero below it.
        scalar_value = 0.5 * (1.0 + std::tanh((point[1] - y_mech_bottom * mesh_size[1]) / epsilon));
      }
  }
  
  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    if (solve_block_id == 0)
      {
        const ScalarValue c   = variable_list.template get_value<Scalar, OldOne>(0);
        const ScalarGrad mux  = variable_list.template get_gradient<Scalar, OldOne>(1);
        const ScalarValue psi = variable_list.template get_value<Scalar, OldOne>(3);
        const ScalarValue phi = variable_list.template get_value<Scalar, OldOne>(4);
        const ScalarGrad psix = variable_list.template get_gradient<Scalar, OldOne>(3);

        const ScalarValue psi_floor = 1.0e-4;
        const number F              = 96485.0;
        const number R              = 8.3145;
        const number j0             = 1.0e3;
        const number T              = 293.15;

        const ScalarValue c_positive = std::max(c,ScalarValue(0.0));

        const ScalarValue M = 2.0 * D_bulk / W * c_positive;
        
        const ScalarValue contact = 2.0 * std::max(c, ScalarValue(0.0));

        const ScalarValue k_bv = contact * j0 * F / (R * T);

        const ScalarValue j_bv = k_bv * phi;
        
        const ScalarValue Jnc = j_bv / (F * c_max);
        
        const ScalarValue eq_c = c + sim_timer.get_timestep() * (M * mux * psix + Jnc * psix.norm()) / psi;

        const ScalarGrad eqx_c = -sim_timer.get_timestep() * M * mux;

        variable_list.set_value_term(0, eq_c);
        variable_list.set_gradient_term(0, eqx_c);
      }
    else if (solve_block_id == 1) // mu
      {
        const ScalarValue c   = variable_list.template get_value<Scalar, Current>(0);
        const ScalarGrad  cx  = variable_list.template get_gradient<Scalar, Current>(0);
        const ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);
        const ScalarGrad psix = variable_list.template get_gradient<Scalar, Current>(3);
        
        ScalarValue fpcV = W / 2.0 * c * (1.0 - c) * (1.0 - 2.0 * c);

        ScalarValue eq_mu  = fpcV - kappa * kappa * cx * psix / psi;
        ScalarGrad  eqx_mu = kappa * kappa * cx;

        variable_list.set_value_term(1, eq_mu);
        variable_list.set_gradient_term(1, eqx_mu);
      }
    else if (solve_block_id == 2) // pp
      {
        const ScalarValue c  = variable_list.template get_value<Scalar, Current>(0);
        const ScalarGrad  cx = variable_list.template get_gradient<Scalar, Current>(0);

        ScalarValue f_tot  = 0.0;
        const ScalarValue f_chem = W / 4.0 * c * c * (1.0 - c) * (1.0 - c);

        const ScalarValue f_grad = 0.5 * kappa * kappa * cx.norm_square();
        f_tot              = f_chem + f_grad;
        variable_list.set_value_term(2, f_tot);
      }
    
      else if (solve_block_id == 4) // electric potential RHS
        {
          const ScalarValue c        = variable_list.template get_value<Scalar, Current>(0);
          const ScalarValue psi      = variable_list.template get_value<Scalar, Current>(3);
          const ScalarValue psi_top  = variable_list.template get_value<Scalar, Current>(5);
          const ScalarGrad  psi_topx = variable_list.template get_gradient<Scalar, Current>(5);

          const ScalarValue contact = 2.0 * std::max(c, ScalarValue(0.0));

          const ScalarValue rhs_phi = -(1.0-psi) * psi_topx.norm() * J_app;

          variable_list.set_value_term(4, rhs_phi);
        }

      else if (solve_block_id == 6) // Mechanics RHS
        {
          const ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);
          const ScalarGrad psi_botx = variable_list.template get_gradient<Scalar, Current>(7);

          VectorValue eq_u = {};

          // Positive pressure at the bottom pushes upward.
          eq_u[1] = p_stack * psi * psi_botx.norm();

          variable_list.set_value_term(6, eq_u);
        }

      else if (solve_block_id == 8) // Stress postprocessing
        {
          VectorGrad strain = variable_list.template get_symmetric_gradient<Vector, Current>(6);
          VectorGrad stress;
          ScalarValue stress_zz(0.0);
          const ScalarValue c = variable_list.template get_value<Scalar, Current>(0);

          const ScalarValue c_mech = std::min(ScalarValue(1.0), std::max(c, ScalarValue(0.0)));

          // Material stiffness interpolation, including residual stiffness.
          const ScalarValue material_weight = elastic_floor + (1.0 - elastic_floor) * c_mech;
          
          PlaneStrain::compute_stress<ScalarValue>(stiffness, strain, stress, stress_zz);

          const ScalarValue sigma_xx = material_weight * stress[0][0];
          const ScalarValue sigma_yy = material_weight * stress[1][1];
          const ScalarValue sigma_xy = material_weight * stress[0][1];

          variable_list.set_value_term(8, sigma_xx);
          variable_list.set_value_term(9, sigma_yy);
          variable_list.set_value_term(10, sigma_xy);
        }
  }
  void
  compute_lhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    if (solve_block_id == 4) // electric potential LHS
      {        
        const ScalarValue phi_lhs = variable_list.template get_value<Scalar, LHS>(4);
        const ScalarGrad phix_lhs = variable_list.template get_gradient<Scalar, LHS>(4);
        const ScalarValue psi     = variable_list.template get_value<Scalar, Current>(3);
        const ScalarValue c       = variable_list.template get_value<Scalar, Current>(0);
        const ScalarGrad psix     = variable_list.template get_gradient<Scalar, Current>(3);
        const ScalarValue psi_top = variable_list.template get_value<Scalar, Current>(5);

        const number sigma       = 0.1;
        const number sigma_floor = 1.0e-6;
        const number F           = 96485.0;
        const number R           = 8.3145;
        const number j0          = 1.0e3;
        const number T           = 293.15;     
        
        // Conductivity with a floor in the Li region so that
        //  the matrix does not become singular.
        const ScalarValue conductivity = sigma * (1.0 - psi) * psi_top + sigma_floor * psi;
        
        const ScalarValue contact = 2.0 * std::max(c, ScalarValue(0.0));

        const ScalarValue k_bv = contact * j0 * F / (R * T);

        const ScalarValue eq_phi = k_bv * phi_lhs * psix.norm() * psi_top;

        const ScalarGrad eqx_phi = conductivity * phix_lhs;

        variable_list.set_value_term(4, eq_phi);
        variable_list.set_gradient_term(4, eqx_phi);
      }

    else if (solve_block_id == 6) // Mechanics LHS
      {
        const VectorValue u = variable_list.template get_value<Vector, LHS>(6);
        const VectorGrad ux = variable_list.template get_gradient<Vector, LHS>(6);
        const ScalarValue c = variable_list.template get_value<Scalar, Current>(0);
        const ScalarValue psi = variable_list.template get_value<Scalar, Current>(3);
        const ScalarGrad psix = variable_list.template get_gradient<Scalar, Current>(3);
        const ScalarValue psi_bot = variable_list.template get_value<Scalar, Current>(7);
        VectorGrad strain = variable_list.template get_symmetric_gradient<Vector, LHS>(6);
        VectorGrad stress;
        ScalarValue stress_zz(0.0);

        const ScalarValue c_mech = std::min(ScalarValue(1.0), std::max(c, ScalarValue(0.0)));

        const number K_n = penalty_factor * E_Li / dx;

        PlaneStrain::compute_stress<ScalarValue>(stiffness, strain, stress, stress_zz);

        // Linear stiffness interpolation in c.
        // A residual stiffness extends the solve into void/exterior regions.
        const ScalarValue solid_weight = psi * psi_bot * c_mech;
        const ScalarValue stiffness_weight = elastic_floor + (1.0 - elastic_floor) * solid_weight;

        VectorGrad lhs_grad_u = stiffness_weight * stress;

        // Normal spring at the Li-SE interface.
        const ScalarValue grad_psi_norm = psix.norm();

        // Units: 1/m. Protect division outside the diffuse interface.
        const ScalarValue grad_floor = ScalarValue(1.0e-6);

        const ScalarValue denominator = std::max(grad_psi_norm, grad_floor);

        const ScalarValue u_dot_grad_psi = u * psix;

        const ScalarValue spring_coefficient = K_n * c_mech * psi_bot / denominator;

        VectorValue lhs_value_u = spring_coefficient * u_dot_grad_psi * psix;

        variable_list.set_value_term(6, lhs_value_u);
        variable_list.set_gradient_term(6, lhs_grad_u);
      }
  }

  number D_bulk;
  number W;
  number gamma;
  number kappa;
  number n_int;
  number J_app;
  number c_max;
  number dx;
  number c_interface_width;
  number psi_interface_width;
  number E_Li;
  number nu_Li;
  number p_stack;
  number penalty_factor;
  number elastic_floor;
  number y_mech_bottom;
  PlaneStrain::VoigtMatrix<number> stiffness;
};

PRISMS_PF_END_NAMESPACE