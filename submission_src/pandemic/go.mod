module github.com/dinvlad/pets-private

go 1.19

replace github.com/ldsec/lattigo/v2 => github.com/hcholab/lattigo/v2 v2.1.2-0.20220628190737-bde274261547

require (
	github.com/aead/chacha20 v0.0.0-20180709150244-8b13a72661da
	github.com/hhcho/frand v1.3.1-0.20210217213629-f1c60c334950
	github.com/hhcho/mpc-core v0.0.0-20210527211839-87c954bf6638
	github.com/ldsec/lattigo/v2 v2.4.0
	go.dedis.ch/onet/v3 v3.2.10
	gonum.org/v1/gonum v0.12.0
)

require (
	github.com/daviddengcn/go-colortext v0.0.0-20180409174941-186a3d44e920 // indirect
	golang.org/x/crypto v0.0.0-20201002170205-7f63de1d35b0 // indirect
	golang.org/x/sys v0.0.0-20200930185726-fdedc70b468f // indirect
	golang.org/x/xerrors v0.0.0-20191011141410-1b5146add898 // indirect
)
