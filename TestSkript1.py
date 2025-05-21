import pytest
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit import transpile as qiskit_transpile

from primitives.circuit import QiskitCircuit, Circuit
from primitives.instruction import QiskitInstruction, Instruction
from primitives.pauli import QiskitPauli, Pauli
from primitives.processor import QiskitProcessor, Processor


def test_qiskit_instruction_and_equality():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    print(qc.data[0])
    print(qc.data[1])
    inst0 = QiskitInstruction(qc.data[0])
    inst1 = QiskitInstruction(qc.data[1])

    # Test weight, __eq__, __hash__, __str__
    assert inst1.weight() == 2
    inst1_copy = QiskitInstruction(qc.data[1])
    assert inst1 == inst1_copy
    assert hash(inst1) == hash(inst1_copy)
    assert "cx" in str(inst1)


def test_qiskit_pauli_basic_operations():
    # pauli creation
    p = QiskitPauli('XZ')
    assert isinstance(p, Pauli)
    label = p.to_label()
    assert label == 'XZ'
    # identity
    id2 = QiskitPauli.ID(2)
    assert id2.to_label() == 'II'

    # multiplication
    p_x = QiskitPauli('X')
    p_z = QiskitPauli('Z')
    p_xz = p_x * p_z
    assert isinstance(p_xz, QiskitPauli)
    assert p_xz.to_label() in 'Y'

    # commutation
    assert p_x.commutes(p_z) == False
    assert p_x.commutes(p_x) == True

    # random and get_composite
    pr = QiskitPauli.random(3)
    assert isinstance(pr, QiskitPauli)
    comp = p_XZ.get_composite(QiskitPauli('ZY')) if False else QiskitPauli('XI').get_composite(QiskitPauli('IX'))
    assert isinstance(comp, QiskitPauli)

    # basis change produces a circuit
    qc = QuantumCircuit(2)
    bc = p.basis_change(QiskitCircuit(qc))
    assert isinstance(bc, QiskitCircuit)
    # ensure operations added
    assert bool(bc)


def test_qiskit_circuit_methods_and_eq_hash():
    qc1 = QuantumCircuit(2)
    circ1 = QiskitCircuit(qc1)
    # add instruction
    inst = QiskitInstruction(qc1.h(0))
    circ1.add_instruction(inst)
    # barrier
    circ1.barrier()
    # measure_all
    circ1.measure_all()
    # inverse and copy_empty
    inv = circ1.inverse()
    assert isinstance(inv, QiskitCircuit)
    ce = circ1.copy_empty()
    assert isinstance(ce, QiskitCircuit)
    # num_qubits and qubits
    assert circ1.num_qubits() == 2
    assert circ1.qubits() == qc1.qubits
    # original
    assert circ1.original() is qc1

    # test compose
    qc2 = QuantumCircuit(2)
    circ2 = QiskitCircuit(qc2)
    circ1.compose(circ2)
    # __getitem__
    _ = circ1[0]
    # __bool__ and __str__
    assert isinstance(str(circ1), str)
    assert bool(circ1) is True

    # __eq__ and __hash__
    circ_copy = QiskitCircuit(qc1)
    assert circ1 == circ_copy
    assert hash(circ1) == hash(circ_copy)


def test_qiskit_processor_submap_and_transpile():
    backend = GenericBackendV2(num_qubits=5)
    proc = QiskitProcessor(backend)
    # sub_map on a subset of qubits
    nodes = list(range(backend.num_qubits))
    sub = proc.sub_map(nodes)
    assert hasattr(sub, 'nodes')
    print(sub)
    assert set(sub.nodes) == set(nodes)

    # transpile wraps into QiskitCircuit
    qc = QuantumCircuit(backend.num_qubits)
    circ = QiskitCircuit(qc)
    transpiled = proc.transpile(circ, inst_map=nodes, optimization_level=0)
    assert isinstance(transpiled, QiskitCircuit)
    assert transpiled.num_qubits() == backend.num_qubits

    # pauli_type property
    assert proc.pauli_type is QiskitPauli
    
    
#test_qiskit_instruction_and_equality()
#test_qiskit_pauli_basic_operations()
#test_qiskit_circuit_methods_and_eq_hash()
test_qiskit_processor_submap_and_transpile()