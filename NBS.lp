\ Model NBS
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  log_uD_dD + log_uO_dO
Subject To
 c_diff_D: - u_D + x_D = -40134.10916073405
 c_diff_O: - u_O + x_O = -1.7412197233040508e+07
 c_uD: - 0.4 zeta_D + 0.4 eta_D_w[0] + 0.4 eta_D_w[1] + 0.4 eta_D_w[2]
   + 0.4 eta_D_w[3] + 0.4 eta_D_w[4] + u_D - 0.12 y_D[0] - 0.12 y_D[1]
   - 0.12 y_D[2] - 0.12 y_D[3] - 0.12 y_D[4] = 0
 c_uO: - 0.4 zeta_O + 0.4 eta_O_w[0] + 0.4 eta_O_w[1] + 0.4 eta_O_w[2]
   + 0.4 eta_O_w[3] + 0.4 eta_O_w[4] + u_O - 0.12 y_O[0] - 0.12 y_O[1]
   - 0.12 y_O[2] - 0.12 y_O[3] - 0.12 y_O[4] = 0
 c_CVaR_D[0]: - zeta_D + eta_D_w[0] + y_D[0] >= 0
 c_CVaR_D[1]: - zeta_D + eta_D_w[1] + y_D[1] >= 0
 c_CVaR_D[2]: - zeta_D + eta_D_w[2] + y_D[2] >= 0
 c_CVaR_D[3]: - zeta_D + eta_D_w[3] + y_D[3] >= 0
 c_CVaR_D[4]: - zeta_D + eta_D_w[4] + y_D[4] >= 0
 c_CVaR_O[0]: - zeta_O + eta_O_w[0] + y_O[0] >= 0
 c_CVaR_O[1]: - zeta_O + eta_O_w[1] + y_O[1] >= 0
 c_CVaR_O[2]: - zeta_O + eta_O_w[2] + y_O[2] >= 0
 c_CVaR_O[3]: - zeta_O + eta_O_w[3] + y_O[3] >= 0
 c_CVaR_O[4]: - zeta_O + eta_O_w[4] + y_O[4] >= 0
 c_yD_link_PaF[0]: y_D[0] + 39878.80882734531 gamma + [
   - 2518.560865711739 S * gamma ] = 39878.80882734531
 c_yD_link_PaF[1]: y_D[1] + 40428.6022591092 gamma + [
   - 2521.266333383127 S * gamma ] = 40428.6022591092
 c_yD_link_PaF[2]: y_D[2] + 40366.96130762107 gamma + [
   - 2522.096002370939 S * gamma ] = 40366.96130762107
 c_yD_link_PaF[3]: y_D[3] + 40470.10875280224 gamma + [
   - 2517.916858447957 S * gamma ] = 40470.10875280225
 c_yD_link_PaF[4]: y_D[4] + 40377.06576808818 gamma + [
   - 2516.408715481073 S * gamma ] = 40377.06576808819
 c_yO_link_PaF[0]: y_O[0] - 39878.80882734545 gamma + [
   2518.560865711739 S * gamma ] = 1.7412144551265389e+07
 c_yO_link_PaF[1]: y_O[1] - 40428.602259109 gamma + [
   2521.266333383127 S * gamma ] = 1.7412448250650283e+07
 c_yO_link_PaF[2]: y_O[2] - 40366.96130762101 gamma + [
   2522.096002370939 S * gamma ] = 1.7412113469887231e+07
 c_yO_link_PaF[3]: y_O[3] - 40470.10875280217 gamma + [
   2517.916858447957 S * gamma ] = 1.7412156907073438e+07
 c_yO_link_PaF[4]: y_O[4] - 40377.06576808825 gamma + [
   2516.408715481073 S * gamma ] = 1.7412402196837116e+07
Bounds
 S <= 150
 x_D >= 1e-06
 x_O >= 1e-06
 gamma <= 1
General Constraints
 c_log_uD_dD: log_uD_dD = LOG ( x_D )
 c_log_uO_dO: log_uO_dO = LOG ( x_O )
End
