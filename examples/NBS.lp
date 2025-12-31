\ Model NBS
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  log_uD_dD + log_uO_dO
Subject To
 c_diff_D: - u_D + x_D = -41546.1779724447
 c_diff_O: - u_O + x_O = -1.4952119616078608e+07
 c_uD: - 0.6 zeta_D + 0.6 eta_D_w[0] + 0.6 eta_D_w[1] + 0.6 eta_D_w[2]
   + 0.6 eta_D_w[3] + u_D - 0.1 y_D[0] - 0.1 y_D[1] - 0.1 y_D[2]
   - 0.1 y_D[3] = 0
 c_uO: - 0.6 zeta_O + 0.6 eta_O_w[0] + 0.6 eta_O_w[1] + 0.6 eta_O_w[2]
   + 0.6 eta_O_w[3] + u_O - 0.1 y_O[0] - 0.1 y_O[1] - 0.1 y_O[2]
   - 0.1 y_O[3] = 0
 c_CVaR_D[0]: - zeta_D + eta_D_w[0] + y_D[0] >= 0
 c_CVaR_D[1]: - zeta_D + eta_D_w[1] + y_D[1] >= 0
 c_CVaR_D[2]: - zeta_D + eta_D_w[2] + y_D[2] >= 0
 c_CVaR_D[3]: - zeta_D + eta_D_w[3] + y_D[3] >= 0
 c_CVaR_O[0]: - zeta_O + eta_O_w[0] + y_O[0] >= 0
 c_CVaR_O[1]: - zeta_O + eta_O_w[1] + y_O[1] >= 0
 c_CVaR_O[2]: - zeta_O + eta_O_w[2] + y_O[2] >= 0
 c_CVaR_O[3]: - zeta_O + eta_O_w[3] + y_O[3] >= 0
 c_yD_link_PaP[0]: y_D[0] + 38071.68679287213 gamma + [
   - 2748.071829585203 S * gamma ] = 38071.68679287213
 c_yD_link_PaP[1]: y_D[1] + 69344.83762438491 gamma + [
   - 2372.507644100579 S * gamma ] = 69344.83762438491
 c_yD_link_PaP[2]: y_D[2] + 41475.7221107704 gamma + [
   - 2772.67949759485 S * gamma ] = 41475.7221107704
 c_yD_link_PaP[3]: y_D[3] + 38139.41243918677 gamma + [
   - 2735.215140368601 S * gamma ] = 38139.41243918677
 c_yO_link_PaP[0]: y_O[0] - 38071.68679287207 gamma + [
   2748.071829585203 S * gamma ] = 1.4982022936340177e+07
 c_yO_link_PaP[1]: y_O[1] - 69344.83762438491 gamma + [
   2372.507644100579 S * gamma ] = 1.4940870126435729e+07
 c_yO_link_PaP[2]: y_O[2] - 41475.72211077035 gamma + [
   2772.67949759485 S * gamma ] = 1.4974787860976003e+07
 c_yO_link_PaP[3]: y_O[3] - 38139.41243918673 gamma + [
   2735.215140368601 S * gamma ] = 1.4978294478419788e+07
Bounds
 S <= 60
 x_D >= 1e-06
 x_O >= 1e-06
 gamma <= 1
General Constraints
 c_log_uD_dD: log_uD_dD = LOG ( x_D )
 c_log_uO_dO: log_uO_dO = LOG ( x_O )
End
