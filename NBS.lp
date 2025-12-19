\ Model NBS
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  log_uD_dD + log_uO_dO
Subject To
 c_diff_D: - u_D + x_D = -8.4934988664562535e+08
 c_diff_O: - u_O + x_O = -4.4906611555544678e+11
 c_uD: - 0.75 zeta_D + 0.75 eta_D_w[0] + 0.75 eta_D_w[1] + 0.75 eta_D_w[2]
   + 0.75 eta_D_w[3] + u_D - 0.0625 y_D[0] - 0.0625 y_D[1] - 0.0625 y_D[2]
   - 0.0625 y_D[3] = 0
 c_uO: u_O - 0.25 y_O[0] - 0.25 y_O[1] - 0.25 y_O[2] - 0.25 y_O[3] = 0
 c_CVaR_D[0]: - zeta_D + eta_D_w[0] + y_D[0] >= 0
 c_CVaR_D[1]: - zeta_D + eta_D_w[1] + y_D[1] >= 0
 c_CVaR_D[2]: - zeta_D + eta_D_w[2] + y_D[2] >= 0
 c_CVaR_D[3]: - zeta_D + eta_D_w[3] + y_D[3] >= 0
 c_CVaR_O[0]: - zeta_O + eta_O_w[0] + y_O[0] >= 0
 c_CVaR_O[1]: - zeta_O + eta_O_w[1] + y_O[1] >= 0
 c_CVaR_O[2]: - zeta_O + eta_O_w[2] + y_O[2] >= 0
 c_CVaR_O[3]: - zeta_O + eta_O_w[3] + y_O[3] >= 0
 c_yD_link_PaP[0]: y_D[0] + 8.1662043119207394e+08 gamma + [
   - 9.6394937212478325e+07 S * gamma ] = 8.1662043119207394e+08
 c_yD_link_PaP[1]: y_D[1] + 1.0092756760549746e+09 gamma + [
   - 8.0365533834166855e+07 S * gamma ] = 1.0092756760549746e+09
 c_yD_link_PaP[2]: y_D[2] + 1.0160601037117224e+09 gamma + [
   - 7.4273716983292863e+07 S * gamma ] = 1.0160601037117224e+09
 c_yD_link_PaP[3]: y_D[3] + 9.4819680106634605e+08 gamma + [
   - 7.6696130370426476e+07 S * gamma ] = 9.4819680106634605e+08
 c_yO_link_PaP[0]: y_O[0] - 8.1662043119209564e+08 gamma + [
   9.6394937212478325e+07 S * gamma ] = 4.4930665117065143e+11
 c_yO_link_PaP[1]: y_O[1] - 1.009275676054975e+09 gamma + [
   8.0365533834166855e+07 S * gamma ] = 4.4904967847917749e+11
 c_yO_link_PaP[2]: y_O[2] - 1.0160601037117318e+09 gamma + [
   7.4273716983292863e+07 S * gamma ] = 4.4889260766082745e+11
 c_yO_link_PaP[3]: y_O[3] - 9.4819680106636012e+08 gamma + [
   7.6696130370426476e+07 S * gamma ] = 4.4901552491113086e+11
Bounds
 S <= 50
 x_D >= 1e-06
 x_O >= 1e-06
 gamma <= 1
General Constraints
 c_log_uD_dD: log_uD_dD = LOG ( x_D )
 c_log_uO_dO: log_uO_dO = LOG ( x_O )
End
