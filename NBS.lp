\ Model NBS
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  log_uD_dD + log_uO_dO
Subject To
 c_diff_D: - u_D + x_D = -40731.88518547544
 c_diff_O: - u_O + x_O = -1.0155872670453716e+07
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
 c_yD_link_PaP[0]: y_D[0] + 35667.76068991201 gamma + [
   - 2605.335589263544 S * gamma ] = 35667.76068991201
 c_yD_link_PaP[1]: y_D[1] + 64780.41709654248 gamma + [
   - 2256.430230592335 S * gamma ] = 64780.41709654248
 c_yD_link_PaP[2]: y_D[2] + 43151.26009833903 gamma + [
   - 2737.782422533915 S * gamma ] = 43151.26009833903
 c_yD_link_PaP[3]: y_D[3] + 39584.60083936193 gamma + [
   - 2693.457994995959 S * gamma ] = 39584.60083936193
 c_yO_link_PaP[0]: y_O[0] - 35667.76068991214 gamma + [
   2605.335589263544 S * gamma ] = 1.0166622495498272e+07
 c_yO_link_PaP[1]: y_O[1] - 64780.4170965421 gamma + [
   2256.430230592335 S * gamma ] = 1.0138696688368859e+07
 c_yO_link_PaP[2]: y_O[2] - 43151.26009833913 gamma + [
   2737.782422533915 S * gamma ] = 1.0161712859445339e+07
 c_yO_link_PaP[3]: y_O[3] - 39584.60083936207 gamma + [
   2693.457994995959 S * gamma ] = 1.016409240831789e+07
Bounds
 S <= 27
 x_D >= 1e-06
 x_O >= 1e-06
 gamma <= 1
General Constraints
 c_log_uD_dD: log_uD_dD = LOG ( x_D )
 c_log_uO_dO: log_uO_dO = LOG ( x_O )
End
