\ Model NBS
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  log_uD_dD + log_uO_dO
Subject To
 c_diff_D: - u_D + x_D = -48218.3286508761
 c_diff_O: - u_O + x_O = -1.329534995857957e+07
 c_uD: - 0.5 zeta_D + 0.5 eta_D_w[0] + 0.5 eta_D_w[1] + 0.5 eta_D_w[2]
   + 0.5 eta_D_w[3] + u_D - 0.125 y_D[0] - 0.125 y_D[1] - 0.125 y_D[2]
   - 0.125 y_D[3] = 0
 c_uO: - 0.1 zeta_O + 0.1 eta_O_w[0] + 0.1 eta_O_w[1] + 0.1 eta_O_w[2]
   + 0.1 eta_O_w[3] + u_O - 0.225 y_O[0] - 0.225 y_O[1] - 0.225 y_O[2]
   - 0.225 y_O[3] = 0
 c_CVaR_D[0]: - zeta_D + eta_D_w[0] + y_D[0] >= 0
 c_CVaR_D[1]: - zeta_D + eta_D_w[1] + y_D[1] >= 0
 c_CVaR_D[2]: - zeta_D + eta_D_w[2] + y_D[2] >= 0
 c_CVaR_D[3]: - zeta_D + eta_D_w[3] + y_D[3] >= 0
 c_CVaR_O[0]: - zeta_O + eta_O_w[0] + y_O[0] >= 0
 c_CVaR_O[1]: - zeta_O + eta_O_w[1] + y_O[1] >= 0
 c_CVaR_O[2]: - zeta_O + eta_O_w[2] + y_O[2] >= 0
 c_CVaR_O[3]: - zeta_O + eta_O_w[3] + y_O[3] >= 0
 c_yD_link_PaP[0]: y_D[0] + 44763.49498489095 gamma + [
   - 2910.558226857347 S * gamma ] = 44763.49498489095
 c_yD_link_PaP[1]: y_D[1] + 68625.23987160246 gamma + [
   - 2350.094973353231 S * gamma ] = 68625.23987160246
 c_yD_link_PaP[2]: y_D[2] + 46018.58894942742 gamma + [
   - 2629.47666122237 S * gamma ] = 46018.58894942742
 c_yD_link_PaP[3]: y_D[3] + 47285.3254615242 gamma + [
   - 2683.811807034024 S * gamma ] = 47285.3254615242
 c_yO_link_PaP[0]: y_O[0] - 44763.49498489055 gamma + [
   2910.558226857347 S * gamma ] = 1.3311680869063856e+07
 c_yO_link_PaP[1]: y_O[1] - 68625.23987160229 gamma + [
   2350.094973353231 S * gamma ] = 1.3271825401504027e+07
 c_yO_link_PaP[2]: y_O[2] - 46018.58894942774 gamma + [
   2629.47666122237 S * gamma ] = 1.3303174276452072e+07
 c_yO_link_PaP[3]: y_O[3] - 47285.32546152419 gamma + [
   2683.811807034024 S * gamma ] = 1.3305174645998565e+07
Bounds
 S <= 40
 x_D >= 1e-06
 x_O >= 1e-06
 gamma <= 1
General Constraints
 c_log_uD_dD: log_uD_dD = LOG ( x_D )
 c_log_uO_dO: log_uO_dO = LOG ( x_O )
End
