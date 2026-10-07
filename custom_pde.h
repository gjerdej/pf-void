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
    , V0(get_user_inputs().user_constants.get_double("V0"))
    , c_max(get_user_inputs().user_constants.get_double("c_max"))
    , kappa(0.0)
    , W(0.0)
    , n_int(get_user_inputs().user_constants.get_double("n_int"))
    , dx(0.0)
    , c_interface_width(0.0)
    , psi_interface_width(0.0)
  {
    const auto &mesh =
      get_user_inputs().spatial_discretization.rectangular_mesh;

    const unsigned int refinement =
      get_user_inputs().spatial_discretization.global_refinement;

    const double refinement_factor =
      std::ldexp(1.0, refinement); // 2^refinement

    const double hx =
      mesh.size[0] /
      (static_cast<double>(mesh.subdivisions[0]) *
      refinement_factor);

    const double hy =
      mesh.size[1] /
      (static_cast<double>(mesh.subdivisions[1]) *
      refinement_factor);

    dx = std::min(hx, hy);

    // Assumes n_int means the number of element spacings.
    c_interface_width = n_int * dx;

    W =
      48.0 * std::log(3) * gamma /
      c_interface_width;

    kappa =
      std::sqrt(
        3.0 * gamma *
        c_interface_width /
        (2.0 * std::log(3)));

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
    const dealii::Tensor<1, dim> &mesh_size =
      get_user_inputs().spatial_discretization.rectangular_mesh.size;

    const std::vector<unsigned int> &mesh_subdivs =
      get_user_inputs().spatial_discretization.rectangular_mesh.subdivisions;

    if (index == 0)
      {
        const double hx =
          mesh_size[0] / static_cast<double>(mesh_subdivs[0]);
        const double hy =
          mesh_size[1] / static_cast<double>(mesh_subdivs[1]);

        // Use the smaller element spacing to define the interface width.
        const double h = std::min(hx, hy);

        const double x0 = 0.5 * mesh_size[0];
        const double y0 = 0.5 * mesh_size[1];
        const double radius = mesh_size[0] / 6.0;

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
        const double epsilon =
          psi_interface_width / (2.0 * std::log(3));

        const double psi_min = 1.0e-6;

        const double profile =
          0.5 * (1.0 - std::tanh(signed_distance / epsilon));

        scalar_value =
          psi_min + (1.0 - psi_min) * profile;
      }

    if (index == 3)
      {
        const double hx =
          mesh_size[0] / static_cast<double>(mesh_subdivs[0]);
        const double hy =
          mesh_size[1] / static_cast<double>(mesh_subdivs[1]);

        // Use the smaller element spacing to define the interface width.
        const double h = std::min(hx, hy);

        const double x0 = 0.5 * mesh_size[0];
        const double y0 = 0.5 * mesh_size[1];
        const double radius = mesh_size[0] / 6.0;

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
        const double epsilon =
          psi_interface_width / (2.0 * std::log(3));

        const double psi_min = 1.0e-6;

        const double profile =
          0.5 * (1.0 - std::tanh(signed_distance / epsilon));

        scalar_value =
          psi_min + (1.0 - psi_min) * profile;
      }
  }

  void
  set_dirichlet([[maybe_unused]] const unsigned int       &index,
                [[maybe_unused]] const unsigned int       &boundary_id,
                [[maybe_unused]] const unsigned int       &component,
                [[maybe_unused]] const dealii::Point<dim> &point,
                [[maybe_unused]] const SimulationTimer    &sim_timer,
                [[maybe_unused]] number                   &scalar_value,
                [[maybe_unused]] number &vector_component_value) const override
  {
    [[maybe_unused]] const double x = (dim > 0) ? point[0] : 0.0;
    [[maybe_unused]] const double y = (dim > 1) ? point[1] : 0.0;
    [[maybe_unused]] const double z = (dim > 2) ? point[2] : 0.0;

    if (index == 4)
      {
      if (boundary_id == RectangularMesh<dim>::Boundary::Bottom)
        {
          scalar_value = V0;
          return;
        }
      }
  }
  
  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    if (solve_block_id == 0)
      {
        const ScalarValue c =
          variable_list.template get_value<Scalar, OldOne>(0);

        const ScalarValue mu =
          variable_list.template get_value<Scalar, OldOne>(1);

        const ScalarGrad mux =
          variable_list.template get_gradient<Scalar, OldOne>(1);

        const ScalarValue psi =
          variable_list.template get_value<Scalar, OldOne>(3);

        const ScalarValue phi =
          variable_list.template get_value<Scalar, OldOne>(4);

        const ScalarGrad psix =
          variable_list.template get_gradient<Scalar, OldOne>(3);

        const ScalarValue psi_floor   = 1.0e-4;
        const number F                = 96485.0;
        const number R                = 8.3145;
        const number j0               = 1.0e3;
        const number T                = 293.15;

        const ScalarValue c_positive =
          std::max(c,ScalarValue(0.0));

        const ScalarValue M =
          2.0 * D_bulk / W * c_positive;
        
        const ScalarValue contact =
          std::max(c, ScalarValue(0.0));

        const ScalarValue j0_eff =
          j0 * std::exp(mu / (2.0 * R * T * c_max));

        const ScalarValue k_bv =
          contact * j0_eff * F / (R * T);

        const ScalarValue j_bv = k_bv * (phi - mu / (F * c_max));
        const ScalarValue Jnc = j_bv / (F * c_max);
        
        const ScalarValue eq_c =
          c + sim_timer.get_timestep() *
            (M * mux * psix + Jnc * psix.norm()) / psi;

        const ScalarGrad eqx_c =
          -sim_timer.get_timestep() * M * mux;

        variable_list.set_value_term(0, eq_c);
        variable_list.set_gradient_term(0, eqx_c);
      }
    else if (solve_block_id == 1) // mu
      {
        ScalarValue c  = variable_list.template get_value<Scalar, Current>(0);
        ScalarGrad  cx = variable_list.template get_gradient<Scalar, Current>(0);

        const ScalarValue psi =
          variable_list.template get_value<Scalar, OldOne>(3);

        const ScalarGrad psix =
          variable_list.template get_gradient<Scalar, OldOne>(3);

        const ScalarValue psi_floor = 1.0e-4;

        const ScalarValue psi_reg =
          std::max(psi,psi_floor);

        ScalarValue fpcV = W / 2.0 * c * (1.0 - c) * (1.0 - 2.0 * c);

        ScalarValue eq_mu  = fpcV - kappa * kappa * cx * psix / psi_reg;
        ScalarGrad  eqx_mu = kappa * kappa * cx;

        variable_list.set_value_term(1, eq_mu);
        variable_list.set_gradient_term(1, eqx_mu);
      }
    else if (solve_block_id == 2) // pp
      {
        ScalarValue c  = variable_list.template get_value<Scalar, Current>(0);
        ScalarGrad  cx = variable_list.template get_gradient<Scalar, Current>(0);

        ScalarValue f_tot  = 0.0;
        const ScalarValue f_chem =
          W / 4.0 *
          c * c * (1.0 - c) * (1.0 - c);

        const ScalarValue f_grad =
          0.5 * kappa * kappa * cx.norm_square();
        f_tot              = f_chem + f_grad;
        variable_list.set_value_term(2, f_tot);
      }
    
      else if (solve_block_id == 4) // electric potential RHS
        {
            const ScalarValue c =
                variable_list.template get_value<Scalar, Current>(0);

            const ScalarValue mu =
                variable_list.template get_value<Scalar, Current>(1);

            const ScalarGrad psix =
                variable_list.template get_gradient<Scalar, Current>(3);

            const number F  = 96485.0;
            const number R  = 8.3145;
            const number j0 = 1.0e3;
            const number T  = 293.15;

            const ScalarValue contact =
                std::max(c, ScalarValue(0.0));

            const ScalarValue j0_eff =
                j0 * std::exp(mu / (2.0 * R * T * c_max));

            const ScalarValue k_bv =
                contact * j0_eff * F / (R * T);

            const ScalarValue rhs_phi =
                k_bv * mu / (F * c_max) * psix.norm();

            variable_list.set_value_term(4, rhs_phi);
        }
  }
  void
  compute_lhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override
  {
    if (solve_block_id == 4) // electric potential LHS
      {
        const ScalarValue mu =
          variable_list.template get_value<Scalar, Current>(1);
        
        const ScalarValue phi_lhs =
          variable_list.template
            get_value<Scalar, LHS>(4);

        const ScalarGrad phix_lhs =
          variable_list.template
            get_gradient<Scalar, LHS>(4);

        const ScalarValue psi =
          variable_list.template
            get_value<Scalar, Current>(3);

        const ScalarValue c =
          variable_list.template
            get_value<Scalar, Current>(0);

        const ScalarGrad psix =
          variable_list.template
            get_gradient<Scalar, Current>(3);

        const number sigma       = 0.1;
        const number sigma_floor = 1.0e-6;
        const number F           = 96485.0;
        const number R           = 8.3145;
        const number j0          = 1.0e3;
        const number T           = 293.15;     
        
                
        // Conductivity with a floor in the Li region so that
        //  the matrix does not become singular.
        const ScalarValue conductivity = sigma * (1.0 - psi) + sigma_floor * psi;
        
        const ScalarValue contact =
          std::max(c, ScalarValue(0.0));

        const ScalarValue j0_eff =
          j0 * std::exp(mu / (2.0 * R * T * c_max));

        const ScalarValue k_bv =
          contact * j0_eff * F / (R * T);

        const ScalarValue eq_phi =
          k_bv * phi_lhs * psix.norm();

        // Ohmic conduction contribution.
        const ScalarGrad eqx_phi =
          conductivity * phix_lhs;

        variable_list.set_value_term(4, eq_phi);
        variable_list.set_gradient_term(4, eqx_phi);
      }
  }

  number D_bulk;
  number W;
  number gamma;
  number kappa;
  number n_int;
  number V0;
  number c_max;
  number dx;
  number c_interface_width;
  number psi_interface_width;
};

PRISMS_PF_END_NAMESPACE
