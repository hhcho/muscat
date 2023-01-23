package mheexample

import (
	"os"
	"strconv"
	"testing"
)

var pid, _ = strconv.Atoi(os.Getenv("PID"))

func TestExampleProtocol(t *testing.T) {
	//cps := mpc.NewCryptoParamsForNetwork(ckks.DefaultParams[ckks.PN14QP438], 2, 20, 2)
	//for i := range cps {
	//	crypto.SaveCryptoParamsAndRotKeys(i+1, cps[i].Sk, cps[i].AggregateSk, cps[i].Pk, cps[i].Rlk, cps[i].RotKs)
	//}
	//log.Fatal()
	exampleProtInstance := InitializeExampleProtocol(pid, "config/")

	exampleProtInstance.ExampleProtocol()

}
