// SPDX-FileCopyrightText: © 2026 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/parse_cmd_options.h>
#include <prismspf/core/problem.h>

using namespace prismspf;

int
main(int argc, char *argv[])
{
  // Initialize MPI
  prismspf::MPIInitFinalize mpi_init(argc, argv);

  // Parse the command line options (if there are any) to get the name of the input
  // file
  ParseCMDOptions cli_options(argc, argv);
  std::string     parameters_filename = cli_options.get_parameters_filename();

  constexpr unsigned int dim    = 2; // TODO change to 3 (original app)
  constexpr unsigned int degree = 1; // TODO change to 1 (original app)

  std::vector<FieldAttributes> fields = {FieldAttributes("c"),
                                         FieldAttributes("mu"),
                                         FieldAttributes("f_tot"),
                                         FieldAttributes("psi"),
                                         FieldAttributes("phi")};

  SolveBlock c_block;
  c_block.id               = 0;
  c_block.solve_type       = Explicit;
  c_block.solve_timing     = Initialized;
  c_block.field_indices    = {0};
  c_block.dependencies_rhs = make_dependency_set(fields, {"old_1(c)", "grad(old_1(c))", "old_1(mu)", "grad(old_1(mu))", "old_1(psi)", "grad(old_1(psi))", "old_1(phi)"});

  SolveBlock mu_block;
  mu_block.id               = 1;
  mu_block.solve_type       = Explicit;
  mu_block.solve_timing     = Uninitialized;
  mu_block.field_indices    = {1};
  mu_block.dependencies_rhs = make_dependency_set(fields, {"c", "grad(c)", "old_1(psi)", "grad(old_1(psi))"});

  SolveBlock pp_block;
  pp_block.id               = 2;
  pp_block.solve_type       = Explicit;
  pp_block.solve_timing     = PostProcess;
  pp_block.field_indices    = {2};
  pp_block.dependencies_rhs = make_dependency_set(fields, {"c", "grad(c)"});

  SolveBlock psi_block;
  psi_block.id            = 3;
  psi_block.solve_type    = Constant;
  psi_block.solve_timing  = Initialized;
  psi_block.field_indices = {3};
  // No RHS evaluation is needed for a field initialized once and held constant.
  psi_block.dependencies_rhs = make_dependency_set(fields, {});

  SolveBlock phi_block;
  phi_block.id              = 4;
  phi_block.solve_type      = Linear;
  phi_block.solve_timing    = Uninitialized;
  phi_block.field_indices   = {4};
  phi_block.dependencies_lhs =
    make_dependency_set(fields,
                        {"lhs(phi)",
                        "grad(lhs(phi))",
                        "psi",
                        "grad(psi)",
                        "c",
                        "mu",
                        "grad(c)"});
  phi_block.dependencies_rhs =
    make_dependency_set(fields, {"c", "mu", "grad(psi)"});


  std::vector<SolveBlock> solve_blocks({c_block, mu_block, pp_block, psi_block, phi_block});


  UserInputParameters<dim>       user_inputs(parameters_filename);
  PhaseFieldTools<dim>           pf_tools;
  CustomPDE<dim, degree, double> pde_operator(user_inputs, pf_tools);
  Problem<dim, degree, double>   problem(fields,
                                       solve_blocks,
                                       user_inputs,
                                       pf_tools,
                                       pde_operator);
  problem.solve();

  return 0;
}
